import xgboost as xgb
import numpy as np


class XGBoostModelEnhanced:
    def __init__(self, device):
        self.device = device
        self.model = None

    def asymmetric_loss_grad_hess(self, y_true, y_pred, alpha=0.7):
        """
        Custom gradient and hessian for asymmetric loss
        alpha > 0.5 penalizes underprediction more
        """
        errors = y_pred - y_true  # Note: XGBoost uses pred - true
        
        # Gradient (first derivative)
        grad = np.where(errors <= 0, 2 * alpha * errors, 2 * (1 - alpha) * errors)
        
        # Hessian (second derivative) - constant for quadratic loss
        hess = np.where(errors <= 0, 2 * alpha, 2 * (1 - alpha))
        
        return grad, hess

    def quantile_loss_grad_hess(self, y_true, y_pred, quantile=0.6):
        """
        Quantile regression loss - targets specific quantile
        quantile > 0.5 helps reduce underprediction
        """
        errors = y_pred - y_true
        
        # Gradient
        grad = np.where(errors <= 0, quantile - 1, quantile)
        
        # Hessian (approximation for quantile loss)
        hess = np.ones_like(errors) * 0.1  # Small constant for numerical stability
        
        return grad, hess

    def train(self, X_train, y_train, params: dict, eval_set: tuple = None, 
              loss_type='default', loss_params=None):
        """
        Enhanced training with custom loss functions
        
        loss_type: 'default', 'asymmetric', or 'quantile'
        loss_params: dict with parameters for custom losses
        """
        if loss_params is None:
            loss_params = {}
            
        # Copy so we don't mutate the caller's dict
        params = params.copy()

        # Set up custom objective if specified
        if loss_type == 'asymmetric':
            alpha = loss_params.get('alpha', 0.7)
            def custom_objective(y_true, y_pred):
                grad, hess = self.asymmetric_loss_grad_hess(y_true, y_pred, alpha)
                return grad, hess
            params['objective'] = custom_objective
            params.pop('eval_metric', None)  # Remove eval_metric for custom objective
            
        elif loss_type == 'quantile':
            quantile = loss_params.get('quantile', 0.6)
            def custom_objective(y_true, y_pred):
                grad, hess = self.quantile_loss_grad_hess(y_true, y_pred, quantile)
                return grad, hess
            params['objective'] = custom_objective
            params.pop('eval_metric', None)
            
        else:
            # Default settings
            params.setdefault('eval_metric', 'rmse')
            
        params.setdefault('verbosity', 0)
        params.setdefault('tree_method', 'gpu_hist')
        params.setdefault('predictor', 'gpu_predictor')

        # Instantiate the model
        self.model = xgb.XGBRegressor(**params)

        if eval_set is not None:
            X_val, y_val = eval_set
            if loss_type in ['asymmetric', 'quantile']:
                # For custom objectives, don't use eval_set to avoid conflicts
                self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


# Alias for backward compatibility
XGBoostModel = XGBoostModelEnhanced