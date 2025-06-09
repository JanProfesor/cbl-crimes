import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class OptimizedPoliceAllocation:
    """
    Enhanced optimized police resource allocation system focused on maximizing
    crime prevention effectiveness while respecting operational constraints.
    """
    
    def __init__(self, base_officers_per_ward: int = 100, 
                 hours_per_officer_per_day: int = 2, 
                 patrol_days_per_week: int = 4):
        """
        Initialize with resource constraints
        
        Args:
            base_officers_per_ward: Number of officers available per ward per day
            hours_per_officer_per_day: Hours each officer can dedicate to burglary patrol
            patrol_days_per_week: Days per week officers patrol for burglary
        """
        self.base_officers_per_ward = base_officers_per_ward
        self.hours_per_officer_per_day = hours_per_officer_per_day
        self.patrol_days_per_week = patrol_days_per_week
        self.daily_hours_per_ward = base_officers_per_ward * hours_per_officer_per_day
        self.weekly_hours_per_ward = self.daily_hours_per_ward * patrol_days_per_week
        
        # Special operations constraints
        self.max_special_ops_officers = 300
        self.special_ops_frequency_months = 4  # Once every 4 months max
        
        print(f"Resource Configuration:")
        print(f"- {self.base_officers_per_ward} officers per ward (MAXIMUM)")
        print(f"- {self.hours_per_officer_per_day} hours per officer per day for burglary patrol")
        print(f"- {self.patrol_days_per_week} patrol days per week")
        print(f"- {self.daily_hours_per_ward} total hours per ward per day (if all officers deployed)")
        print(f"- {self.weekly_hours_per_ward} total hours per ward per week (if all officers deployed)")
    
    def load_and_preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Enhanced data loading with better risk categorization"""
        df = pd.read_csv(csv_path)
        
        # Create more sophisticated risk categories using statistical methods
        # Use interquartile range for outlier detection
        Q1 = df['pred_ensemble'].quantile(0.25)
        Q3 = df['pred_ensemble'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define thresholds
        low_threshold = Q1
        medium_threshold = Q3
        high_threshold = Q3 + 0.5 * IQR
        critical_threshold = Q3 + 1.5 * IQR  # Statistical outlier threshold
        
        def get_enhanced_risk_category(pred):
            if pred <= low_threshold:
                return 'Low'
            elif pred <= medium_threshold:
                return 'Medium'
            elif pred <= high_threshold:
                return 'High'
            else:
                return 'Critical'
        
        df['risk_category'] = df['pred_ensemble'].apply(get_enhanced_risk_category)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        
        # Add seasonal and temporal features
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        
        # Calculate z-scores for standardized risk assessment
        df['risk_zscore'] = (df['pred_ensemble'] - df['pred_ensemble'].mean()) / df['pred_ensemble'].std()
        
        print(f"\nData loaded: {len(df)} records, {df['ward_code'].nunique()} unique wards")
        print(f"Risk distribution:")
        print(df['risk_category'].value_counts().sort_index())
        print(f"\nRisk thresholds:")
        print(f"- Low: ≤ {low_threshold:.2f}")
        print(f"- Medium: ≤ {medium_threshold:.2f}")
        print(f"- High: ≤ {high_threshold:.2f}")
        print(f"- Critical: > {high_threshold:.2f}")
        print(f"\nCONSTRAINT: All allocations will be capped at {self.base_officers_per_ward} officers per ward")
        
        return df
    
    def calculate_efficiency_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sophisticated efficiency weights considering multiple factors
        """
        df = df.copy()
        
        # Base weight using logarithmic scaling for diminishing returns
        df['base_weight'] = np.log1p(df['pred_ensemble'])
        
        # Seasonal adjustment factor
        seasonal_multipliers = {'Winter': 1.2, 'Spring': 1.0, 'Summer': 0.9, 'Autumn': 1.1}
        df['seasonal_factor'] = df['season'].map(seasonal_multipliers)
        
        # Risk tier multiplier (exponential for critical cases)
        risk_multipliers = {'Low': 0.5, 'Medium': 1.0, 'High': 1.8, 'Critical': 3.2}
        df['risk_multiplier'] = df['risk_category'].map(risk_multipliers)
        
        # Z-score adjustment for extreme outliers
        df['zscore_adjustment'] = np.where(df['risk_zscore'] > 2, 1.5, 1.0)
        
        # Combined efficiency weight
        df['efficiency_weight'] = (df['base_weight'] * 
                                 df['seasonal_factor'] * 
                                 df['risk_multiplier'] * 
                                 df['zscore_adjustment'])
        
        # Normalize to prevent extreme allocations
        df['efficiency_weight'] = np.clip(df['efficiency_weight'], 0.1, 10.0)
        
        return df
    
    def optimized_allocation_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced optimized allocation using multi-factor efficiency weights
        """
        df = self.calculate_efficiency_weights(df)
        monthly_allocations = []
        
        for (year, month), group in df.groupby(['year', 'month']):
            group = group.copy()
            
            # Calculate available resources for this month
            total_wards = len(group)
            total_weekly_hours = total_wards * self.weekly_hours_per_ward
            
            # Allocate based on efficiency weights
            total_weight = group['efficiency_weight'].sum()
            
            if total_weight > 0:
                group['allocated_weekly_hours'] = (group['efficiency_weight'] / total_weight) * total_weekly_hours
                group['allocated_daily_hours'] = group['allocated_weekly_hours'] / self.patrol_days_per_week
                group['allocated_officers'] = group['allocated_daily_hours'] / self.hours_per_officer_per_day
            else:
                # Fallback to equal allocation
                group['allocated_weekly_hours'] = self.weekly_hours_per_ward
                group['allocated_daily_hours'] = self.daily_hours_per_ward
                group['allocated_officers'] = self.base_officers_per_ward
            
            # CRITICAL: Enforce maximum constraint - cannot exceed 100 officers per ward
            group['allocated_officers'] = np.minimum(group['allocated_officers'], self.base_officers_per_ward)
            
            # Ensure minimum viable patrol presence (at least 20% of base allocation)
            min_officers = self.base_officers_per_ward * 0.2
            group['allocated_officers'] = np.maximum(group['allocated_officers'], min_officers)
            
            # Recalculate hours based on constrained officer allocation
            group['allocated_daily_hours'] = group['allocated_officers'] * self.hours_per_officer_per_day
            group['allocated_weekly_hours'] = group['allocated_daily_hours'] * self.patrol_days_per_week
            
            group['allocation_method'] = 'Optimized'
            monthly_allocations.append(group)
        
        result = pd.concat(monthly_allocations, ignore_index=True)
        
        # Calculate comprehensive efficiency metrics
        result = self.calculate_efficiency_metrics(result)
        
        return result
    
    def calculate_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple efficiency metrics for the allocation"""
        df = df.copy()
        
        # Prediction-allocation correlation
        pred_alloc_corr = np.corrcoef(df['pred_ensemble'], df['allocated_officers'])[0, 1]
        if np.isnan(pred_alloc_corr):
            pred_alloc_corr = 0
        
        # Resource concentration (coefficient of variation)
        resource_concentration = df['allocated_officers'].std() / df['allocated_officers'].mean()
        
        # High-risk coverage efficiency
        high_risk_mask = df['risk_category'].isin(['High', 'Critical'])
        high_risk_officers = df[high_risk_mask]['allocated_officers'].sum()
        total_officers = df['allocated_officers'].sum()
        high_risk_coverage = high_risk_officers / total_officers if total_officers > 0 else 0
        
        # Prevention effectiveness score (weighted by prediction accuracy)
        df['prevention_score'] = df['allocated_officers'] * df['pred_ensemble']
        total_prevention = df['prevention_score'].sum()
        max_possible = df['pred_ensemble'].sum() * df['allocated_officers'].max()
        prevention_efficiency = total_prevention / max_possible if max_possible > 0 else 0
        
        # Combined efficiency score
        efficiency_score = (pred_alloc_corr * 0.3 + 
                          high_risk_coverage * 0.4 + 
                          prevention_efficiency * 0.3)
        
        df['efficiency_score'] = efficiency_score
        df['prediction_correlation'] = pred_alloc_corr
        df['resource_concentration'] = resource_concentration
        df['high_risk_coverage'] = high_risk_coverage
        df['prevention_efficiency'] = prevention_efficiency
        
        return df
    
    def identify_special_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced special operations identification with operational constraints
        """
        # Identify candidates (top 2% of predictions)
        threshold = df['pred_ensemble'].quantile(0.98)
        candidates = df[df['pred_ensemble'] >= threshold].copy()
        
        if len(candidates) == 0:
            return pd.DataFrame()
        
        # Sort by prediction and date to prioritize
        candidates = candidates.sort_values(['pred_ensemble', 'year', 'month'], ascending=[False, True, True])
        
        # Apply frequency constraint (max once every 4 months per ward)
        selected_operations = []
        ward_last_operation = {}
        
        for _, row in candidates.iterrows():
            ward = row['ward_code']
            current_date = datetime(int(row['year']), int(row['month']), 1)
            
            # Check if enough time has passed since last operation
            if ward in ward_last_operation:
                months_since = (current_date.year - ward_last_operation[ward].year) * 12 + \
                              (current_date.month - ward_last_operation[ward].month)
                if months_since < self.special_ops_frequency_months:
                    continue
            
            # Calculate additional officers needed
            pred_value = row['pred_ensemble']
            base_additional = min(int(pred_value * 2), 150)  # 2 officers per predicted burglary, max 150
            
            # Scale based on risk level
            if row['risk_category'] == 'Critical':
                additional_officers = min(base_additional * 2, self.max_special_ops_officers)
                operation_type = 'Critical Intervention'
            else:
                additional_officers = min(base_additional, self.max_special_ops_officers)
                operation_type = 'Enhanced Patrol'
            
            # Calculate operation duration (1-4 weeks based on severity)
            if pred_value >= df['pred_ensemble'].quantile(0.995):
                duration_weeks = 4
            elif pred_value >= df['pred_ensemble'].quantile(0.99):
                duration_weeks = 3
            elif pred_value >= df['pred_ensemble'].quantile(0.985):
                duration_weeks = 2
            else:
                duration_weeks = 1
            
            operation = {
                'ward_code': ward,
                'year': int(row['year']),
                'month': int(row['month']),
                'pred_ensemble': pred_value,
                'risk_category': row['risk_category'],
                'additional_officers': additional_officers,
                'total_officers': self.base_officers_per_ward + additional_officers,
                'operation_type': operation_type,
                'duration_weeks': duration_weeks,
                'estimated_cost': additional_officers * 40 * duration_weeks,  # £40/hour estimate
                'priority_score': pred_value * (1 + additional_officers / 100)
            }
            
            selected_operations.append(operation)
            ward_last_operation[ward] = current_date
        
        if not selected_operations:
            return pd.DataFrame()
        
        special_ops_df = pd.DataFrame(selected_operations)
        special_ops_df = special_ops_df.sort_values('priority_score', ascending=False)
        
        return special_ops_df
    
    def generate_optimization_report(self, df: pd.DataFrame, special_ops: pd.DataFrame) -> Dict:
        """Generate comprehensive optimization report"""
        
        # Overall statistics
        total_wards = df['ward_code'].nunique()
        total_months = len(df.groupby(['year', 'month']))
        avg_prediction = df['pred_ensemble'].mean()
        
        # Risk distribution
        risk_dist = df['risk_category'].value_counts(normalize=True) * 100
        
        # Resource allocation statistics
        avg_officers = df['allocated_officers'].mean()
        officer_std = df['allocated_officers'].std()
        min_officers = df['allocated_officers'].min()
        max_officers = df['allocated_officers'].max()
        
        # Efficiency metrics
        efficiency_scores = df.iloc[0]  # Since all rows have same efficiency scores
        
        # Special operations summary
        if len(special_ops) > 0:
            total_special_ops = len(special_ops)
            total_additional_officers = special_ops['additional_officers'].sum()
            avg_duration = special_ops['duration_weeks'].mean()
            total_cost = special_ops['estimated_cost'].sum()
        else:
            total_special_ops = 0
            total_additional_officers = 0
            avg_duration = 0
            total_cost = 0
        
        # Seasonal analysis
        seasonal_stats = df.groupby('season').agg({
            'pred_ensemble': 'mean',
            'allocated_officers': 'mean'
        }).round(2)
        
        # Convert to nested dictionary for easier access
        seasonal_dict = {}
        for season in seasonal_stats.index:
            seasonal_dict[season] = {
                'pred_ensemble': seasonal_stats.loc[season, 'pred_ensemble'],
                'allocated_officers': seasonal_stats.loc[season, 'allocated_officers']
            }
        
        report = {
            'summary': {
                'total_wards': total_wards,
                'total_ward_months': len(df),
                'analysis_months': total_months,
                'avg_predicted_burglaries': round(avg_prediction, 2)
            },
            'risk_distribution': risk_dist.to_dict(),
            'resource_allocation': {
                'avg_officers_per_ward': round(avg_officers, 1),
                'allocation_std_dev': round(officer_std, 1),
                'min_allocation': round(min_officers, 1),
                'max_allocation': round(max_officers, 1),
                'allocation_range_ratio': round(max_officers / min_officers, 2)
            },
            'efficiency_metrics': {
                'overall_efficiency': round(efficiency_scores['efficiency_score'], 3),
                'prediction_correlation': round(efficiency_scores['prediction_correlation'], 3),
                'high_risk_coverage': round(efficiency_scores['high_risk_coverage'], 3),
                'prevention_efficiency': round(efficiency_scores['prevention_efficiency'], 3),
                'resource_concentration': round(efficiency_scores['resource_concentration'], 3)
            },
            'special_operations': {
                'total_operations': total_special_ops,
                'additional_officers_needed': total_additional_officers,
                'avg_operation_duration_weeks': round(avg_duration, 1),
                'estimated_total_cost_gbp': total_cost
            },
            'seasonal_analysis': seasonal_dict
        }
        
        return report
    
    def run_optimization(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run the complete optimization process
        
        Returns:
            Tuple of (allocation_df, special_ops_df, report_dict)
        """
        print("=" * 60)
        print("ENHANCED OPTIMIZED POLICE RESOURCE ALLOCATION")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_path)
        
        # Run optimized allocation
        print("\nRunning optimized allocation strategy...")
        allocation_result = self.optimized_allocation_strategy(df)
        
        # Identify special operations
        print("\nIdentifying special operations...")
        special_ops = self.identify_special_operations(df)
        
        # Generate report
        print("\nGenerating optimization report...")
        report = self.generate_optimization_report(allocation_result, special_ops)
        
        # Print summary
        self.print_optimization_summary(report, special_ops)
        
        return allocation_result, special_ops, report
    
    def print_optimization_summary(self, report: Dict, special_ops: pd.DataFrame):
        """Print formatted optimization summary"""
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\nDATA OVERVIEW:")
        print(f"- Total wards analyzed: {report['summary']['total_wards']:,}")
        print(f"- Total ward-months: {report['summary']['total_ward_months']:,}")
        print(f"- Average predicted burglaries per ward-month: {report['summary']['avg_predicted_burglaries']}")
        
        print(f"\nRISK DISTRIBUTION:")
        for risk, pct in report['risk_distribution'].items():
            print(f"- {risk}: {pct:.1f}%")
        
        print(f"\nRESOURCE ALLOCATION:")
        print(f"- Average officers per ward: {report['resource_allocation']['avg_officers_per_ward']}")
        print(f"- Allocation range: {report['resource_allocation']['min_allocation']:.1f} - {report['resource_allocation']['max_allocation']:.1f}")
        print(f"- Allocation efficiency ratio: {report['resource_allocation']['allocation_range_ratio']:.2f}x")
        
        print(f"\nEFFICIENCY METRICS:")
        print(f"- Overall efficiency score: {report['efficiency_metrics']['overall_efficiency']:.3f}")
        print(f"- Prediction correlation: {report['efficiency_metrics']['prediction_correlation']:.3f}")
        print(f"- High-risk coverage: {report['efficiency_metrics']['high_risk_coverage']:.1%}")
        print(f"- Prevention efficiency: {report['efficiency_metrics']['prevention_efficiency']:.3f}")
        
        print(f"\nSPECIAL OPERATIONS:")
        print(f"- Operations recommended: {report['special_operations']['total_operations']}")
        print(f"- Additional officers needed: {report['special_operations']['additional_officers_needed']}")
        print(f"- Average operation duration: {report['special_operations']['avg_operation_duration_weeks']:.1f} weeks")
        print(f"- Estimated total cost: £{report['special_operations']['estimated_total_cost_gbp']:,}")
        
        print(f"\nSEASONAL INSIGHTS:")
        for season, stats in report['seasonal_analysis'].items():
            print(f"- {season}: {stats['pred_ensemble']:.2f} avg predictions, {stats['allocated_officers']:.1f} avg officers")
        
        if len(special_ops) > 0:
            print(f"\nTOP 5 SPECIAL OPERATIONS:")
            top_ops = special_ops.head(5)
            for _, op in top_ops.iterrows():
                print(f"- Ward {op['ward_code']} ({op['year']}-{op['month']:02d}): {op['additional_officers']} officers, {op['operation_type']}")
        
        print(f"\nRECOMMENDATIONS:")
        print("✓ Deploy optimized allocation for maximum prevention effectiveness")
        print("✓ Implement special operations for highest-risk situations")
        print("✓ Monitor seasonal patterns for resource planning")
        print("✓ Regular review and adjustment based on actual outcomes")


def main():
    """Main execution function"""
    
    # Initialize the allocation system
    allocator = OptimizedPoliceAllocation(
        base_officers_per_ward=100,
        hours_per_officer_per_day=2,
        patrol_days_per_week=4
    )
    
    # Run optimization
    allocation_df, special_ops_df, report = allocator.run_optimization('quick_fix_test_predictions.csv')
    
    # Save results
    print(f"\nSaving results...")
    allocation_df.to_csv("optimized_allocation_enhanced.csv", index=False)
    if len(special_ops_df) > 0:
        special_ops_df.to_csv("special_operations_enhanced.csv", index=False)
    
    # Save detailed report
    import json
    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✓ Enhanced optimized allocation saved to optimized_allocation_enhanced.csv")
    print("✓ Special operations saved to special_operations_enhanced.csv")
    print("✓ Detailed report saved to optimization_report.json")
    
    return allocation_df, special_ops_df, report


if __name__ == "__main__":
    allocation_df, special_ops_df, report = main()