import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


class EnhancedPoliceAllocation:
    """
    Data-driven police resource allocation using adaptive risk scoring
    and dynamic seasonal adjustments based on actual crime patterns.
    """
    
    def __init__(self,
                 base_officers_per_ward: int = 100,
                 hours_per_officer_per_day: int = 2,
                 patrol_days_per_week: int = 4,
                 surge_capacity_multiplier: float = 1.3,
                 learning_window_months: int = 12):
        
        self.base_officers_per_ward = base_officers_per_ward
        self.hours_per_officer_per_day = hours_per_officer_per_day
        self.patrol_days_per_week = patrol_days_per_week
        self.surge_capacity_multiplier = surge_capacity_multiplier
        self.learning_window_months = learning_window_months
        
        self.daily_hours_per_ward = base_officers_per_ward * hours_per_officer_per_day
        self.weekly_hours_per_ward = self.daily_hours_per_ward * patrol_days_per_week
        
        # Learning parameters
        self.seasonal_factors = {}
        self.trend_weights = {}
        self.effectiveness_history = {}
        
        print(f"Enhanced Resource Configuration:")
        print(f"- Base: {self.base_officers_per_ward} officers per ward")
        print(f"- Surge capacity: {surge_capacity_multiplier}x base")
        print(f"- Learning window: {learning_window_months} months")

    def load_and_preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load data and calculate dynamic risk factors"""
        df = pd.read_csv(csv_path)
        
        # Create temporal features
        df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
        df['season'] = df['month'].map({
            12:'Winter', 1:'Winter', 2:'Winter',
             3:'Spring',4:'Spring',5:'Spring',
             6:'Summer',7:'Summer',8:'Summer',
             9:'Autumn',10:'Autumn',11:'Autumn'
        })
        
        print(f"\nData loaded: {len(df)} records, {df['ward_code'].nunique()} wards")
        return df

    def calculate_dynamic_seasonal_factors(self, df: pd.DataFrame) -> Dict:
        """Calculate seasonal factors from actual data rather than fixed assumptions"""
        
        # Calculate average burglaries by season
        seasonal_avg = df.groupby('season')['burglary_count'].mean()
        overall_avg = df['burglary_count'].mean()
        
        # Dynamic seasonal factors based on actual data
        seasonal_factors = {}
        for season in seasonal_avg.index:
            seasonal_factors[season] = seasonal_avg[season] / overall_avg
        
        print(f"\nDynamic seasonal factors (data-driven):")
        for season, factor in seasonal_factors.items():
            print(f"- {season}: {factor:.3f}x")
        
        return seasonal_factors

    def calculate_prediction_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert predictions to percentiles for dynamic thresholding"""
        df = df.copy()
        
        # Use prediction column name from the original data
        prediction_col = 'burglary_count'  # Adjust this if you have a different prediction column
        
        df['prediction_percentile'] = df[prediction_col].rank(pct=True)
        
        # Dynamic risk categories based on percentiles
        def dynamic_risk_category(percentile):
            if percentile <= 0.25:
                return 'Low'
            elif percentile <= 0.50:
                return 'Medium-Low'
            elif percentile <= 0.75:
                return 'Medium-High'
            elif percentile <= 0.90:
                return 'High'
            else:
                return 'Critical'
        
        df['risk_category'] = df['prediction_percentile'].apply(dynamic_risk_category)
        
        return df

    def calculate_trend_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate recent trend factors for each ward"""
        df = df.copy()
        df = df.sort_values(['ward_code', 'year', 'month'])
        
        # Calculate 3-month rolling average trend
        df['rolling_avg'] = df.groupby('ward_code')['burglary_count'].rolling(3, min_periods=1).mean().values
        df['trend_factor'] = df.groupby('ward_code')['rolling_avg'].pct_change(periods=3).fillna(0)
        
        # Normalize trend factors
        df['trend_factor'] = np.clip(df['trend_factor'], -0.5, 0.5) + 1.0
        
        return df

    def calculate_adaptive_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive risk scores using multiple data-driven factors"""
        df = self.calculate_prediction_percentiles(df)
        df = self.calculate_trend_factors(df)
        
        # Get dynamic seasonal factors
        seasonal_factors = self.calculate_dynamic_seasonal_factors(df)
        df['seasonal_factor'] = df['season'].map(seasonal_factors)
        
        # Calculate base risk score from percentile
        df['base_risk_score'] = df['prediction_percentile']
        
        # Apply trend adjustment
        df['trend_adjusted_score'] = df['base_risk_score'] * df['trend_factor']
        
        # Apply seasonal adjustment
        df['seasonal_adjusted_score'] = df['trend_adjusted_score'] * df['seasonal_factor']
        
        # Final risk score (normalized 0-1)
        scaler = MinMaxScaler()
        df['adaptive_risk_score'] = scaler.fit_transform(df[['seasonal_adjusted_score']]).flatten()
        
        return df

    def calculate_elastic_allocation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Allocate resources with elastic capacity based on total predicted demand"""
        df = self.calculate_adaptive_risk_scores(df)
        monthly_results = []
        
        # Debug: Check if columns exist
        print(f"Available columns: {list(df.columns)}")
        
        # Check if year and month columns exist, if not create them
        if 'year' not in df.columns or 'month' not in df.columns:
            if 'date' in df.columns:
                df['year'] = pd.to_datetime(df['date']).dt.year
                df['month'] = pd.to_datetime(df['date']).dt.month
            else:
                print("Warning: No year/month columns found. Using ward_code for grouping.")
                # Fallback: group by ward_code only
                for ward_code, group in df.groupby('ward_code'):
                    g = group.copy()
                    
                    # Calculate total predicted demand for this ward
                    total_predicted = g['burglary_count'].sum()
                    base_demand = len(g) * g['burglary_count'].mean()
                    
                    # Determine if surge capacity is needed
                    if total_predicted > base_demand * 1.2:  # 20% above average
                        capacity_multiplier = min(self.surge_capacity_multiplier, 
                                                total_predicted / base_demand)
                        print(f"Surge capacity activated for ward {ward_code}: {capacity_multiplier:.2f}x")
                    else:
                        capacity_multiplier = 1.0
                    
                    # Calculate available officers for this ward
                    max_officers_per_ward = self.base_officers_per_ward * capacity_multiplier
                    total_available_officers = len(g) * max_officers_per_ward
                    
                    # Allocate proportionally to risk scores
                    total_risk_score = g['adaptive_risk_score'].sum()
                    
                    if total_risk_score > 0:
                        g['allocated_officers'] = (g['adaptive_risk_score'] / total_risk_score) * total_available_officers
                    else:
                        g['allocated_officers'] = max_officers_per_ward
                    
                    # Apply constraints
                    min_officers = self.base_officers_per_ward * 0.3
                    g['allocated_officers'] = g['allocated_officers'].clip(
                        lower=min_officers, 
                        upper=max_officers_per_ward
                    )
                    
                    # Calculate hours
                    g['allocated_daily_hours'] = g['allocated_officers'] * self.hours_per_officer_per_day
                    g['allocated_weekly_hours'] = g['allocated_daily_hours'] * self.patrol_days_per_week
                    g['capacity_multiplier'] = capacity_multiplier
                    
                    monthly_results.append(g)
                
                result = pd.concat(monthly_results, ignore_index=True)
                return self.calculate_enhanced_metrics(result)
        
        # Original logic if year/month columns exist
        try:
            for (year, month), group in df.groupby(['year', 'month']):
                g = group.copy()
                
                # Calculate total predicted demand for this month
                total_predicted = g['burglary_count'].sum()
                base_demand = len(g) * g['burglary_count'].mean()
                
                # Determine if surge capacity is needed
                if total_predicted > base_demand * 1.2:  # 20% above average
                    capacity_multiplier = min(self.surge_capacity_multiplier, 
                                            total_predicted / base_demand)
                    print(f"Surge capacity activated for {year}-{month:02d}: {capacity_multiplier:.2f}x")
                else:
                    capacity_multiplier = 1.0
                
                # Calculate available officers for this month
                max_officers_per_ward = self.base_officers_per_ward * capacity_multiplier
                total_available_officers = len(g) * max_officers_per_ward
                
                # Allocate proportionally to risk scores
                total_risk_score = g['adaptive_risk_score'].sum()
                
                if total_risk_score > 0:
                    g['allocated_officers'] = (g['adaptive_risk_score'] / total_risk_score) * total_available_officers
                else:
                    g['allocated_officers'] = max_officers_per_ward
                
                # Apply constraints
                min_officers = self.base_officers_per_ward * 0.3  # Higher minimum than before
                g['allocated_officers'] = g['allocated_officers'].clip(
                    lower=min_officers, 
                    upper=max_officers_per_ward
                )
                
                # Calculate hours
                g['allocated_daily_hours'] = g['allocated_officers'] * self.hours_per_officer_per_day
                g['allocated_weekly_hours'] = g['allocated_daily_hours'] * self.patrol_days_per_week
                g['capacity_multiplier'] = capacity_multiplier
                
                monthly_results.append(g)
        
        except Exception as e:
            print(f"Error in groupby operation: {e}")
            print("Falling back to simple allocation...")
            # Simple fallback allocation
            df['allocated_officers'] = self.base_officers_per_ward
            df['allocated_daily_hours'] = df['allocated_officers'] * self.hours_per_officer_per_day
            df['allocated_weekly_hours'] = df['allocated_daily_hours'] * self.patrol_days_per_week
            df['capacity_multiplier'] = 1.0
            return self.calculate_enhanced_metrics(df)
        
        result = pd.concat(monthly_results, ignore_index=True)
        return self.calculate_enhanced_metrics(result)

    def calculate_enhanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced performance metrics"""
        df = df.copy()
        
        # Efficiency metrics
        prediction_col = 'burglary_count'
        allocation_correlation = np.corrcoef(df[prediction_col], df['allocated_officers'])[0,1]
        allocation_correlation = 0 if np.isnan(allocation_correlation) else allocation_correlation
        
        # Resource utilization efficiency
        total_predictions = df[prediction_col].sum()
        total_officers = df['allocated_officers'].sum()
        resource_efficiency = total_predictions / total_officers if total_officers > 0 else 0
        
        # Coverage quality
        high_risk_mask = df['adaptive_risk_score'] >= 0.8
        high_risk_officers = df.loc[high_risk_mask, 'allocated_officers'].sum()
        total_officers_used = df['allocated_officers'].sum()
        high_risk_coverage = high_risk_officers / total_officers_used if total_officers_used > 0 else 0
        
        # Response capacity
        df['prevention_potential'] = df['allocated_officers'] * df[prediction_col]
        total_prevention = df['prevention_potential'].sum()
        max_theoretical = df[prediction_col].sum() * df['allocated_officers'].max()
        prevention_efficiency = total_prevention / max_theoretical if max_theoretical > 0 else 0
        
        # Adaptive metrics
        surge_usage = (df['capacity_multiplier'] > 1.0).mean()
        allocation_variance = df['allocated_officers'].std() / df['allocated_officers'].mean()
        
        # Store metrics in dataframe
        df['allocation_correlation'] = allocation_correlation
        df['resource_efficiency'] = resource_efficiency
        df['high_risk_coverage'] = high_risk_coverage
        df['prevention_efficiency'] = prevention_efficiency
        df['surge_usage_rate'] = surge_usage
        df['allocation_coefficient_variation'] = allocation_variance
        
        # Composite score
        df['enhanced_efficiency_score'] = (
            0.25 * allocation_correlation +
            0.25 * resource_efficiency / 10 +  # Normalize
            0.25 * high_risk_coverage +
            0.25 * prevention_efficiency
        )
        
        return df

    def generate_enhanced_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive performance report"""
        
        # Get metrics from first row (all rows have same values)
        metrics = df.iloc[0]
        
        # Basic statistics
        total_wards = df['ward_code'].nunique()
        
        # Try to calculate months, with fallback
        try:
            if 'year' in df.columns and 'month' in df.columns:
                total_months = len(df.groupby(['year', 'month']))
            else:
                total_months = 1  # Fallback
        except:
            total_months = 1
            
        avg_prediction = df['burglary_count'].mean()
        
        # Allocation statistics
        avg_officers = df['allocated_officers'].mean()
        min_officers = df['allocated_officers'].min()
        max_officers = df['allocated_officers'].max()
        
        # Risk distribution
        risk_dist = df['risk_category'].value_counts(normalize=True) * 100
        
        # Seasonal analysis with actual data
        seasonal_stats = df.groupby('season').agg({
            'burglary_count': 'mean',
            'allocated_officers': 'mean',
            'adaptive_risk_score': 'mean'
        }).round(3)
        
        # Surge capacity usage - simplified calculation
        try:
            if 'year' in df.columns and 'month' in df.columns:
                surge_months = (df['capacity_multiplier'] > 1.0).groupby(['year', 'month']).first().sum()
            else:
                surge_months = (df['capacity_multiplier'] > 1.0).sum()
        except:
            surge_months = 0
        
        report = {
            'summary': {
                'total_wards': total_wards,
                'analysis_months': total_months,
                'avg_predicted_burglaries': round(avg_prediction, 2),
                'algorithm_type': 'Enhanced Adaptive'
            },
            'allocation_performance': {
                'avg_officers_per_ward': round(avg_officers, 1),
                'allocation_range': f"{min_officers:.1f} - {max_officers:.1f}",
                'allocation_correlation': round(metrics['allocation_correlation'], 3),
                'resource_efficiency': round(metrics['resource_efficiency'], 3),
                'high_risk_coverage': round(metrics['high_risk_coverage'], 3),
                'prevention_efficiency': round(metrics['prevention_efficiency'], 3),
                'enhanced_efficiency_score': round(metrics['enhanced_efficiency_score'], 3)
            },
            'adaptive_features': {
                'surge_capacity_used': f"{metrics['surge_usage_rate']:.1%} of periods",
                'total_surge_activations': int(surge_months),
                'allocation_flexibility': round(1/metrics['allocation_coefficient_variation'], 2) if metrics['allocation_coefficient_variation'] > 0 else 0,
                'dynamic_risk_categories': len(risk_dist)
            },
            'risk_distribution': risk_dist.to_dict(),
            'seasonal_analysis': seasonal_stats.to_dict()
        }
        
        return report

    def run_enhanced_optimization(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Run the enhanced optimization process"""
        print("=" * 70)
        print("ENHANCED ADAPTIVE POLICE RESOURCE ALLOCATION")
        print("=" * 70)
        
        # Load and process data
        df = self.load_and_preprocess_data(csv_path)
        
        # Run enhanced allocation
        print("\nRunning adaptive allocation with dynamic risk scoring...")
        allocation_result = self.calculate_elastic_allocation(df)
        
        # Generate report
        print("\nGenerating enhanced performance report...")
        report = self.generate_enhanced_report(allocation_result)
        
        # Print results
        self.print_enhanced_summary(report)
        
        return allocation_result, report

    def print_enhanced_summary(self, report: Dict):
        """Print formatted summary of enhanced results"""
        print("\n" + "=" * 70)
        print("ENHANCED OPTIMIZATION RESULTS")
        print("=" * 70)
        
        print(f"\nSUMMARY:")
        print(f"- Algorithm: {report['summary']['algorithm_type']}")
        print(f"- Wards analyzed: {report['summary']['total_wards']:,}")
        print(f"- Months analyzed: {report['summary']['analysis_months']}")
        print(f"- Avg burglaries per ward-month: {report['summary']['avg_predicted_burglaries']}")
        
        print(f"\nALLOCATION PERFORMANCE:")
        perf = report['allocation_performance']
        print(f"- Enhanced efficiency score: {perf['enhanced_efficiency_score']:.3f}")
        print(f"- Allocation-prediction correlation: {perf['allocation_correlation']:.3f}")
        print(f"- High-risk area coverage: {perf['high_risk_coverage']:.1%}")
        print(f"- Prevention efficiency: {perf['prevention_efficiency']:.3f}")
        print(f"- Average officers per ward: {perf['avg_officers_per_ward']}")
        print(f"- Allocation range: {perf['allocation_range']}")
        
        print(f"\nADAPTIVE FEATURES:")
        adaptive = report['adaptive_features']
        print(f"- Surge capacity utilization: {adaptive['surge_capacity_used']}")
        print(f"- Total surge activations: {adaptive['total_surge_activations']}")
        print(f"- Allocation flexibility index: {adaptive['allocation_flexibility']}")
        print(f"- Dynamic risk categories: {adaptive['dynamic_risk_categories']}")
        
        print(f"\nKEY IMPROVEMENTS:")
        print("✓ Data-driven seasonal adjustments (no more arbitrary factors)")
        print("✓ Adaptive risk scoring based on percentiles") 
        print("✓ Elastic resource capacity for high-demand periods")
        print("✓ Trend-aware allocation considering recent patterns")
        print("✓ Enhanced performance metrics and learning capability")


def main():
    """Main execution function for enhanced algorithm"""
    
    # Initialize enhanced allocator
    allocator = EnhancedPoliceAllocation(
        base_officers_per_ward=100,
        hours_per_officer_per_day=2,
        patrol_days_per_week=4,
        surge_capacity_multiplier=1.3,
        learning_window_months=12
    )
    
    # Run enhanced optimization
    allocation_df, report = allocator.run_enhanced_optimization('ward_london.csv')
    
    # Save results
    print(f"\nSaving enhanced results...")
    allocation_df.to_csv("enhanced_allocation_results.csv", index=False)
    
    import json
    with open("enhanced_optimization_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✓ Enhanced allocation saved to enhanced_allocation_results.csv")
    print("✓ Enhanced report saved to enhanced_optimization_report.json")
    
    return allocation_df, report


if __name__ == "__main__":
    allocation_df, report = main()