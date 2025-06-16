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
    Modified to work with realistic_detailed_predictions.csv using 'actual' column.
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
        
        # Use 'actual' column instead of 'burglary_count'
        df['burglary_count'] = df['actual']
        
        print(f"\nData loaded: {len(df)} records, {df['ward_code'].nunique()} wards")
        print(f"Time period: {df['year'].min()}-{df['year'].max()}")
        print(f"Actual burglary statistics:")
        print(f"- Min: {df['actual'].min()}")
        print(f"- Max: {df['actual'].max()}")
        print(f"- Mean: {df['actual'].mean():.2f}")
        print(f"- Std: {df['actual'].std():.2f}")
        
        return df

    def calculate_dynamic_seasonal_factors(self, df: pd.DataFrame) -> Dict:
        """Calculate seasonal factors from actual data rather than fixed assumptions"""
        
        # Calculate average burglaries by season using actual data
        seasonal_avg = df.groupby('season')['actual'].mean()
        overall_avg = df['actual'].mean()
        
        # Dynamic seasonal factors based on actual data
        seasonal_factors = {}
        for season in seasonal_avg.index:
            seasonal_factors[season] = seasonal_avg[season] / overall_avg
        
        print(f"\nDynamic seasonal factors (data-driven from actual crimes):")
        for season, factor in seasonal_factors.items():
            print(f"- {season}: {factor:.3f}x")
        
        return seasonal_factors

    def calculate_prediction_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert actual crime counts to percentiles for dynamic thresholding"""
        df = df.copy()
        
        # Use actual column for percentile calculation
        df['prediction_percentile'] = df['actual'].rank(pct=True)
        
        # Dynamic risk categories based on percentiles of actual crime data
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
        """Calculate recent trend factors for each ward based on actual crime data"""
        df = df.copy()
        df = df.sort_values(['ward_code', 'year', 'month'])
        
        # Calculate 3-month rolling average trend using actual data
        df['rolling_avg'] = df.groupby('ward_code')['actual'].rolling(3, min_periods=1).mean().values
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
        """Allocate resources with elastic capacity based on actual crime demand"""
        df = self.calculate_adaptive_risk_scores(df)
        monthly_results = []
        
        print(f"Calculating elastic allocation based on actual crime data...")
        
        for (year, month), group in df.groupby(['year', 'month']):
            g = group.copy()
            
            # Calculate total actual demand for this month
            total_actual = g['actual'].sum()
            base_demand = len(g) * g['actual'].mean()
            
            # Determine if surge capacity is needed based on actual crime levels
            if total_actual > base_demand * 1.2:  # 20% above average
                capacity_multiplier = min(self.surge_capacity_multiplier, 
                                        total_actual / base_demand)
                print(f"Surge capacity activated for {year}-{month:02d}: {capacity_multiplier:.2f}x (actual crimes: {total_actual:.0f})")
            else:
                capacity_multiplier = 1.0
            
            # Calculate available officers for this month
            max_officers_per_ward = self.base_officers_per_ward * capacity_multiplier
            total_available_officers = len(g) * max_officers_per_ward
            
            # Allocate proportionally to risk scores (based on actual crime patterns)
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

    def calculate_enhanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced performance metrics based on actual crime data"""
        df = df.copy()
        
        # Efficiency metrics - correlation between actual crime and officer allocation
        allocation_correlation = np.corrcoef(df['actual'], df['allocated_officers'])[0,1]
        allocation_correlation = 0 if np.isnan(allocation_correlation) else allocation_correlation
        
        # Resource utilization efficiency - actual crimes per officer
        total_actual_crimes = df['actual'].sum()
        total_officers = df['allocated_officers'].sum()
        resource_efficiency = total_actual_crimes / total_officers if total_officers > 0 else 0
        
        # Coverage quality - focus on high-risk areas
        high_risk_mask = df['adaptive_risk_score'] >= 0.8
        high_risk_officers = df.loc[high_risk_mask, 'allocated_officers'].sum()
        total_officers_used = df['allocated_officers'].sum()
        high_risk_coverage = high_risk_officers / total_officers_used if total_officers_used > 0 else 0
        
        # Response capacity - prevention potential based on actual crime patterns
        df['prevention_potential'] = df['allocated_officers'] * df['actual']
        total_prevention = df['prevention_potential'].sum()
        max_theoretical = df['actual'].sum() * df['allocated_officers'].max()
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
            0.25 * min(resource_efficiency / 10, 1) +  # Normalize and cap
            0.25 * high_risk_coverage +
            0.25 * prevention_efficiency
        )
        
        return df

    def generate_enhanced_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive performance report based on actual crime data"""
        
        # Get metrics from first row (all rows have same values)
        metrics = df.iloc[0]
        
        # Basic statistics
        total_wards = df['ward_code'].nunique()
        
        # Safe calculation of total months - check if year/month columns exist
        try:
            if 'year' in df.columns and 'month' in df.columns:
                total_months = len(df.groupby(['year', 'month']))
            else:
                # Fallback: try to extract from date column or use unique dates
                if 'date' in df.columns:
                    total_months = df['date'].dt.to_period('M').nunique()
                else:
                    total_months = len(df) // total_wards  # Approximate
        except:
            total_months = len(df) // total_wards if total_wards > 0 else 1
            
        avg_actual_crimes = df['actual'].mean()
        total_actual_crimes = df['actual'].sum()
        
        # Allocation statistics
        avg_officers = df['allocated_officers'].mean()
        min_officers = df['allocated_officers'].min()
        max_officers = df['allocated_officers'].max()
        
        # Risk distribution
        risk_dist = df['risk_category'].value_counts(normalize=True) * 100
        
        # Seasonal analysis with actual crime data
        seasonal_stats = df.groupby('season').agg({
            'actual': 'mean',
            'allocated_officers': 'mean',
            'adaptive_risk_score': 'mean'
        }).round(3)
        
        # Surge capacity usage - safe calculation
        try:
            if 'year' in df.columns and 'month' in df.columns:
                surge_months = (df['capacity_multiplier'] > 1.0).groupby(['year', 'month']).first().sum()
            else:
                surge_months = (df['capacity_multiplier'] > 1.0).sum()
        except:
            surge_months = (df['capacity_multiplier'] > 1.0).sum()
        
        # Ward-level analysis
        ward_stats = df.groupby('ward_code').agg({
            'actual': ['mean', 'sum', 'std'],
            'allocated_officers': 'mean',
            'adaptive_risk_score': 'mean'
        }).round(3)
        
        # Top 10 highest crime wards
        top_crime_wards = df.groupby('ward_code')['actual'].sum().nlargest(10)
        
        report = {
            'summary': {
                'total_wards': total_wards,
                'analysis_months': total_months,
                'total_actual_crimes': int(total_actual_crimes),
                'avg_actual_crimes_per_ward_month': round(avg_actual_crimes, 2),
                'algorithm_type': 'Enhanced Adaptive (Actual Crime Data)'
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
                'allocation_flexibility_index': round(1/metrics['allocation_coefficient_variation'], 2) if metrics['allocation_coefficient_variation'] > 0 else 0,
                'dynamic_risk_categories': len(risk_dist)
            },
            'risk_distribution': risk_dist.to_dict(),
            'seasonal_analysis': seasonal_stats.to_dict(),
            'top_crime_wards': top_crime_wards.to_dict()
        }
        
        return report

    def run_enhanced_optimization(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Run the enhanced optimization process"""
        print("=" * 70)
        print("ENHANCED ADAPTIVE POLICE RESOURCE ALLOCATION")
        print("Using Actual Crime Data for Resource Allocation")
        print("=" * 70)
        
        # Load and process data
        df = self.load_and_preprocess_data(csv_path)
        
        # Run enhanced allocation
        print("\nRunning adaptive allocation with dynamic risk scoring based on actual crimes...")
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
        print("ENHANCED OPTIMIZATION RESULTS (ACTUAL CRIME DATA)")
        print("=" * 70)
        
        print(f"\nSUMMARY:")
        print(f"- Algorithm: {report['summary']['algorithm_type']}")
        print(f"- Wards analyzed: {report['summary']['total_wards']:,}")
        print(f"- Months analyzed: {report['summary']['analysis_months']}")
        print(f"- Total actual crimes: {report['summary']['total_actual_crimes']:,}")
        print(f"- Avg crimes per ward-month: {report['summary']['avg_actual_crimes_per_ward_month']}")
        
        print(f"\nALLOCATION PERFORMANCE:")
        perf = report['allocation_performance']
        print(f"- Enhanced efficiency score: {perf['enhanced_efficiency_score']:.3f}")
        print(f"- Allocation-crime correlation: {perf['allocation_correlation']:.3f}")
        print(f"- High-risk area coverage: {perf['high_risk_coverage']:.1%}")
        print(f"- Prevention efficiency: {perf['prevention_efficiency']:.3f}")
        print(f"- Resource efficiency (crimes/officer): {perf['resource_efficiency']:.3f}")
        print(f"- Average officers per ward: {perf['avg_officers_per_ward']}")
        print(f"- Allocation range: {perf['allocation_range']}")
        
        print(f"\nADAPTIVE FEATURES:")
        adaptive = report['adaptive_features']
        print(f"- Surge capacity utilization: {adaptive['surge_capacity_used']}")
        print(f"- Total surge activations: {adaptive['total_surge_activations']}")
        print(f"- Allocation flexibility index: {adaptive['allocation_flexibility_index']}")
        print(f"- Dynamic risk categories: {adaptive['dynamic_risk_categories']}")
        
        print(f"\nTOP 5 HIGHEST CRIME WARDS:")
        top_wards = list(report['top_crime_wards'].items())[:5]
        for i, (ward, crimes) in enumerate(top_wards, 1):
            print(f"{i}. Ward {ward}: {crimes:.0f} total crimes")
        
        print(f"\nKEY IMPROVEMENTS:")
        print("✓ Allocation based on ACTUAL crime data (not predictions)")
        print("✓ Data-driven seasonal adjustments from real crime patterns")
        print("✓ Adaptive risk scoring based on actual crime percentiles") 
        print("✓ Elastic resource capacity for high-crime periods")
        print("✓ Ward-specific trend analysis using historical crime data")
        print("✓ Enhanced performance metrics tied to actual crime prevention")


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
    
    # Run enhanced optimization on the realistic predictions data
    allocation_df, report = allocator.run_enhanced_optimization('realistic_detailed_predictions.csv')
    
    # Save results
    print(f"\nSaving enhanced results...")
    allocation_df.to_csv("enhanced_allocation_results_actual.csv", index=False)
    
    import json
    with open("enhanced_optimization_report_actual.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✓ Enhanced allocation saved to enhanced_allocation_results_actual.csv")
    print("✓ Enhanced report saved to enhanced_optimization_report_actual.json")
    
    # Display some sample allocation results
    print(f"\nSAMPLE ALLOCATION RESULTS:")
    print("=" * 50)
    sample_df = allocation_df[['ward_code', 'year', 'month', 'actual', 'risk_category', 
                              'allocated_officers', 'allocated_daily_hours']].head(10)
    print(sample_df.to_string(index=False))
    
    return allocation_df, report


if __name__ == "__main__":
    allocation_df, report = main()