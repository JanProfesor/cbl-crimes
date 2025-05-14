# Requirements:
# pip install pandas scikit-learn torch pytorch-tabnet optuna tqdm

import os
import json
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetRegressor
from tqdm.auto import tqdm
import warnings
import time
from datetime import datetime, timedelta

# Suppress TabNet warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="pytorch_tabnet.abstract_model")


class ParamSampler:
    @staticmethod
    def sample_random_params(config):
        return {
            "n_d": int(np.random.choice(config["n_d"])),
            "n_steps": int(np.random.choice(config["n_steps"])),
            "gamma": float(np.random.choice(config["gamma"])),  # Changed from uniform to choice
            "lr": float(10 ** np.random.uniform(
                np.log10(config["lr"][0]), np.log10(config["lr"][1])
            )),
            "batch_size": int(np.random.choice(config["batch_size"]))
        }

    @staticmethod
    def suggest_params(trial, config):
        return {
            "n_d": int(trial.suggest_categorical("n_d", config["n_d"])),
            "n_steps": int(trial.suggest_int("n_steps",
                config["n_steps"][0], config["n_steps"][1])),
            "gamma": float(trial.suggest_categorical("gamma", config["gamma"])),  # Changed to categorical
            "lr": float(trial.suggest_loguniform("lr",
                config["lr"][0], config["lr"][1])),
            "batch_size": int(trial.suggest_categorical(
                "batch_size", config["batch_size"]))
        }


class TabNetTrainer:
    def __init__(self, X_train, y_train, X_val, y_val,
                 device, cat_idxs, cat_dims, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.config = config
        
    def train(self, trial, params, trial_idx, total_trials):
        # Instantiate model with modified parameters
        model = TabNetRegressor(
            n_d=params['n_d'], n_a=params['n_d'], n_steps=params['n_steps'],
            gamma=params['gamma'], cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims, cat_emb_dim=1,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": params['lr'], "weight_decay": 1e-5},  # Added weight decay
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,  # Changed scheduler
            scheduler_params={"mode": "min", "factor": 0.5, "patience": 5},  # Adaptive LR
            lambda_sparse=1e-3,  # Regularization parameter
            mask_type='sparsemax', device_name=self.device,
            verbose=0  # Disable TabNet's default logging
        )
        
        # Initialize with artificially high RMSE to prevent first epoch bias
        best_rmse = float('inf')
        no_improve = 0
        max_epochs = self.config['max_epochs']
        best_epoch = 0
        
        # Initialize metric histories for tracking
        rmse_history = []
        mae_history = []
        r2_history = []
        
        # Trial header with nicer formatting
        print(f"\n{'='*80}")
        print(f"⚡ Trial {trial_idx+1}/{total_trials} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Parameters:")
        print(f"  n_d: {params['n_d']}")
        print(f"  n_steps: {params['n_steps']}")
        print(f"  gamma: {params['gamma']:.4f}")
        print(f"  lr: {params['lr']:.6f}")
        print(f"  batch_size: {params['batch_size']}")
        print(f"{'-'*80}")
        
        # Single progress bar for the entire training process
        print(f"Training progress for {max_epochs} epochs:")
        
        # Epoch loop with custom progress tracking
        start_time = time.time()
        progress_fmt = "Epoch {:3d}/{:3d} [{}] ({:.2f}s)"
        
        # Save metrics for each epoch to help track progress
        epoch_metrics = []
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Silence output during fit by redirecting stdout temporarily
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                # Fit one epoch
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)], eval_name=['val'],
                    eval_metric=['rmse'], max_epochs=1, patience=0,
                    batch_size=params['batch_size'], virtual_batch_size=128,
                    drop_last=False
                )
            finally:
                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout
            
            # Evaluate
            preds = model.predict(self.X_val)
            rmse = mean_squared_error(self.y_val, preds)**0.5
            mae = mean_absolute_error(self.y_val, preds)
            r2 = r2_score(self.y_val, preds)
            
            # Store metrics for tracking
            rmse_history.append(rmse)
            mae_history.append(mae)
            r2_history.append(r2)
            
            # Store metrics for display
            epoch_metrics.append({
                'epoch': epoch + 1,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Calculate progress bar
            progress_percent = (epoch + 1) / max_epochs
            bar_width = 50
            bar_filled = int(bar_width * progress_percent)
            bar = '█' * bar_filled + '░' * (bar_width - bar_filled)
            
            # Print progress line with current epoch metrics
            progress_line = progress_fmt.format(
                epoch + 1, max_epochs, bar, epoch_time
            )
            metrics_display = f" RMSE: {rmse:.4f}"
            
            # Show improvement indicator
            if epoch > 0:
                if rmse < rmse_history[-2]:
                    metrics_display += " ↓"  # Improvement
                elif rmse > rmse_history[-2]:
                    metrics_display += " ↑"  # Worse
                else:
                    metrics_display += " ="  # Same
                    
            print(f"{progress_line} {metrics_display}", end='\r')
            
            # Early stopping logic
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch + 1
                no_improve = 0
                # Save the best model state
                # Note: TabNet doesn't have a direct state_dict method, so we'll use the full model
            else:
                no_improve += 1
                if no_improve >= self.config['patience']:
                    print()  # New line
                    time_elapsed = time.time() - start_time
                    print(f"\n🛑 Early stopping at epoch {epoch+1} (no improvement for {self.config['patience']} epochs)")
                    print(f"⏱️  Training time: {timedelta(seconds=int(time_elapsed))}")
                    
                    # Show best metrics
                    print(f"Best RMSE: {best_rmse:.4f} at epoch {best_epoch}")
                    break
        else:
            print()  # New line
            time_elapsed = time.time() - start_time
            print(f"\n✅ Training completed ({max_epochs} epochs)")
            print(f"⏱️  Training time: {timedelta(seconds=int(time_elapsed))}")
            print(f"Best RMSE: {best_rmse:.4f} at epoch {best_epoch}")

        # Display training curve summary
        if len(rmse_history) > 1:
            print("\nTraining progression:")
            max_display = min(5, len(rmse_history))
            for idx in range(max_display):
                e = epoch_metrics[idx]
                print(f"Epoch {e['epoch']:2d}: RMSE={e['rmse']:.4f}, MAE={e['mae']:.4f}, R²={e['r2']:.4f}")
            
            if len(rmse_history) > max_display + 2:
                print("...")
                
            # Show final epochs
            if len(rmse_history) > max_display:
                for idx in range(-min(3, len(rmse_history) - max_display), 0):
                    e = epoch_metrics[idx]
                    print(f"Epoch {e['epoch']:2d}: RMSE={e['rmse']:.4f}, MAE={e['mae']:.4f}, R²={e['r2']:.4f}")
        
        # Final metrics
        final_preds = model.predict(self.X_val)
        final_rmse = mean_squared_error(self.y_val, final_preds)**0.5
        final_mae = mean_absolute_error(self.y_val, final_preds)
        final_r2 = r2_score(self.y_val, final_preds)
        
        # Report to Optuna
        trial.report(best_rmse, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        return best_rmse, {
            "rmse": best_rmse, 
            "mae": final_mae, 
            "r2": final_r2,
            "history": {
                "rmse": rmse_history,
                "mae": mae_history,
                "r2": r2_history
            },
            "best_epoch": best_epoch
        }, model

    def test(self, model):
        preds = model.predict(self.X_val)
        return mean_squared_error(self.y_val, preds)**0.5


class OptunaBayesHyperband:
    def __init__(self, config, trainer, output_dir="optuna_results"):
        self.config = config
        self.trainer = trainer
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.pruner = HyperbandPruner(
            min_resource=config['min_epochs'],
            max_resource=config['max_epochs'],
            reduction_factor=config['eta']
        )
        self.sampler = TPESampler()
        self.study = optuna.create_study(
            direction="minimize", sampler=self.sampler, pruner=self.pruner
        )
        self.results = []
        self.start_time = None

    def objective(self, trial):
        idx = trial.number
        total = self.config['n_trials']
        
        # Determine if this is a random trial or Optuna-suggested
        if idx < self.config['random_trials']:
            params = ParamSampler.sample_random_params(self.config)
            trial_type = "RANDOM"
        else:
            params = ParamSampler.suggest_params(trial, self.config)
            trial_type = "OPTUNA"
            
        # Track time for this trial
        trial_start = time.time()
        
        # Train the model
        rmse, metrics, model = self.trainer.train(trial, params, idx, total)
        
        # Test the model
        test_rmse = self.trainer.test(model)
        
        # Save the model
        path = os.path.join(self.output_dir, f"model_trial_{idx}.zip")
        model.save_model(path)
        
        # Calculate elapsed time
        trial_time = time.time() - trial_start
        total_time = time.time() - self.start_time
        
        # Update remaining time estimate
        trials_done = idx + 1
        trials_left = total - trials_done
        avg_time_per_trial = total_time / trials_done
        est_time_left = avg_time_per_trial * trials_left
        
        # Store results
        result = {
            'trial': idx,
            'trial_type': trial_type,
            'params': params,
            'rmse': rmse,
            'test_rmse': test_rmse,
            'metrics': metrics,
            'trial_time': trial_time,
            'best_epoch': metrics.get('best_epoch', 0)
        }
        self.results.append(result)
        
        # Save results after each trial
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
            
        # Print progress summary
        print(f"\n{'='*80}")
        print(f"Trial {trials_done}/{total} completed - {trial_type}")
        print(f"Best RMSE: {rmse:.6f} at epoch {metrics.get('best_epoch', 0)}")
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Trial time: {timedelta(seconds=int(trial_time))}")
        print(f"Total elapsed: {timedelta(seconds=int(total_time))}")
        print(f"Est. remaining: {timedelta(seconds=int(est_time_left))}")
        print(f"Est. completion: {(datetime.now() + timedelta(seconds=est_time_left)).strftime('%H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Report to Optuna
        trial.report(rmse, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        return rmse

    def run(self):
        print("\n" + "="*80)
        print(f"🚀 Starting TabNet Hyperparameter Optimization")
        print(f"Total trials: {self.config['n_trials']} (Random: {self.config['random_trials']}, Optuna: {self.config['n_trials']-self.config['random_trials']})")
        print("="*80 + "\n")
        
        # Start timing
        self.start_time = time.time()
        
        # Create progress bar for all trials
        with tqdm(total=self.config['n_trials'], 
                  desc="Hyperparameter Optimization", 
                  unit="trial",
                  position=0, leave=True) as pbar:
                  
            # Define callback to update progress bar
            def progress_callback(study, trial):
                pbar.update(1)
                
            # Run optimization
            self.study.optimize(
                self.objective, 
                n_trials=self.config['n_trials'],
                callbacks=[progress_callback]
            )
        
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Print final summary
        best = self.study.best_trial
        best_result = next(r for r in self.results if r['trial']==best.number)
        
        print("\n" + "="*80)
        print("🎉 OPTIMIZATION COMPLETED")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print("="*80)
        
        print("\n🏆 BEST TRIAL")
        print(f"Trial number: {best.number}")
        print(f"Trial type: {best_result['trial_type']}")
        print(f"Best epoch: {best_result.get('best_epoch', 0)}")
        print(f"Best RMSE: {best.value:.6f}")
        print(f"Test RMSE: {best_result['test_rmse']:.6f}")
        print(f"Best parameters:")
        for param, value in (best.params.items() if hasattr(best, 'params') else best_result['params'].items()):
            print(f"  {param}: {value}")
        
        # Create a summary file with full details
        summary = {
            'best_trial': best.number,
            'best_params': best.params if hasattr(best, 'params') else best_result['params'],
            'best_metrics': {
                'rmse': best.value,
                'mae': best_result['metrics']['mae'],
                'r2': best_result['metrics']['r2'],
                'test_rmse': best_result['test_rmse'],
                'best_epoch': best_result.get('best_epoch', 0)
            },
            'total_time': total_time,
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save detailed results to file
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"\nDetailed results saved to: {self.output_dir}")
        print(f"Best model saved at: {os.path.join(self.output_dir, f'model_trial_{best.number}.zip')}")


if __name__ == "__main__":
    # Configuration with improved parameters for better training
    config = {
        'n_d': [16, 32, 64],  # Network dimensions
        'n_steps': [3, 5, 7],  # Number of decision steps
        'gamma': [1.0, 1.5, 2.0],  # Relaxation parameter
        'lr': [1e-3, 1e-2],  # Learning rate range
        'batch_size': [128, 256, 512],  # Batch sizes
        'min_epochs': 5,  # Minimum epochs
        'max_epochs': 30,  # Increased max epochs for better convergence
        'eta': 3,  # Hyperband parameter
        'random_trials': 1,  # Random trials
        'n_trials': 5,  # Total trials
        'patience': 7  # Increased patience for better convergence
    }
    
    print("📊 TabNet Hyperparameter Optimization")
    print(f"Loading dataset...")
    
    # Load data with progress indicator
    try:
        df = pd.read_csv("processed/final_dataset_all.csv")
        print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        exit(1)
    
    # Data preprocessing
    print("Preprocessing data...")
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.sort_values(['ward_code', 'date'])

    TARGET = 'burglary_count'

    # Add lag features
    for lag in (1, 2, 3, 6):
        df[f'{TARGET}_lag_{lag}'] = df.groupby('ward_code')[TARGET].shift(lag)

    # Add rolling mean features
    for window in [3, 6]:
        df[f'{TARGET}_rolling_mean_{window}'] = df.groupby('ward_code')[TARGET].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())

    # Add trend features
    df['month_num'] = df['date'].dt.month
    df['year_num'] = df['date'].dt.year

    # One-hot encode month for seasonality
    month_dummies = pd.get_dummies(df['month_num'], prefix='month')
    df = pd.concat([df, month_dummies], axis=1)

    # Drop all rows with ANY NaN values - this is the important change
    print(f"Shape before dropping NaN values: {df.shape}")
    df = df.dropna()
    print(f"Shape after dropping NaN values: {df.shape}")

    # Split features and target
    X = df.drop(columns=[TARGET, 'date'])
    y = df[TARGET].values.reshape(-1, 1)

    # Convert categorical columns to numeric before checking for NaNs
    X['ward_code'] = X['ward_code'].astype('category').cat.codes
    cat_idxs = [X.columns.get_loc('ward_code')]
    cat_dims = [X['ward_code'].nunique()]

    # Fix: Convert DataFrame to NumPy array and check for NaN values using pandas instead of numpy
    print(f"NaN values in X: {pd.isna(X).sum().sum()}")
    print(f"NaN values in y: {pd.isna(pd.DataFrame(y)).sum().sum()}")

    # Train-val split
    print("Splitting data into train and validation sets...")
    # Convert to numpy arrays after ensuring no NaN values exist
    X_numpy = X.to_numpy().astype(np.float32)  # Convert to a compatible numeric type
    y_numpy = y.astype(np.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_numpy, y_numpy, test_size=0.3, random_state=42)

    # Check for NaNs after split
    print(f"NaN values in X_train: {np.isnan(X_train).sum()}")
    print(f"NaN values in X_val: {np.isnan(X_val).sum()}")
    print(f"NaN values in y_train: {np.isnan(y_train).sum()}")
    print(f"NaN values in y_val: {np.isnan(y_val).sum()}")
    
    # Setup device with proper CUDA optimization
    if torch.cuda.is_available():
        device = "cuda"
        # Print GPU information
        cuda_device = torch.cuda.current_device()
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(cuda_device)}")
        print(f"    CUDA Version: {torch.version.cuda}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.2f} GB")
        
        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory optimization
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print(f"🖥️  GPU not available, using CPU")
    
    # Setup trainer and optimizer
    trainer = TabNetTrainer(X_train, y_train, X_val, y_val,
                            device, cat_idxs, cat_dims, config)
    search = OptunaBayesHyperband(config, trainer)
    
    # Run optimization
    search.run()