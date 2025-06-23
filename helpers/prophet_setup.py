import pandas as pd
from prophet import Prophet

class ParcelPredictor:
    def __init__(self):
        self.file_path = 'data/prophet_train.xlsx'
        self.sheets = {
            'parcel_count': 'ParcelCount',
            'fleet_available': 'FleetAvailable',
            'total_parcel_weight': 'TotalParcelWeight',
            'avg_parcel_weight': 'AvgParcelWeight',
            'avg_parcel_volume_size': 'AvgParcelVolumeSize',
        }

        self.holidays_df = pd.read_excel(self.file_path, sheet_name='Holidays')
        self.holidays_df['ds'] = pd.to_datetime(self.holidays_df['ds'])
        self.merged_df = None
        self.date_df = None

    def forecast_feature(self, days, sheet_name):
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        model = Prophet(holidays=self.holidays_df, growth='flat')
        model.fit(df)

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        last_date = df['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_date]

        if sheet_name in ['ParcelCount', 'FleetAvailable']:
            future_forecast.loc[:, 'yhat'] = future_forecast['yhat'].round().astype(int)

        return future_forecast[['ds', 'yhat']]

    def process_data(self, days):
        for feature, sheet in self.sheets.items():
            pred = self.forecast_feature(days, sheet)
            pred.rename(columns={'yhat': feature}, inplace=True)

            if self.merged_df is None:
                self.merged_df = pred
            else:
                self.merged_df = pd.merge(self.merged_df, pred, on='ds', how='outer')

        self.date_df = self.merged_df.copy()
        self.date_df['year'] = self.date_df['ds'].dt.year
        self.date_df['month'] = self.date_df['ds'].dt.month
        self.date_df['day'] = self.date_df['ds'].dt.day
        self.date_df['day_of_week'] = self.date_df['ds'].dt.day_name()
        self.date_df['is_weekend'] = self.date_df['day_of_week'].isin(['Saturday', 'Sunday']).map({True: 'Yes', False: 'No'})

        holiday_patterns = self.holidays_df['ds'].dt.strftime('%m-%d').unique()
        self.date_df['is_holiday'] = self.date_df['ds'].dt.strftime('%m-%d').isin(holiday_patterns).astype(int)
        self.date_df['is_holiday_soon'] = 0
        
        for days_before in range(1, 5):
            self.date_df['is_holiday_soon'] |= self.date_df['ds'].shift(-days_before).dt.strftime('%m-%d').isin(holiday_patterns).fillna(0).astype(int)

        self.date_df['fleet_available_3'] = self.date_df['fleet_available']
        self.date_df = self.date_df[['year', 'month', 'day', 'parcel_count', 'day_of_week', 'is_weekend',
                           'is_holiday', 'is_holiday_soon', 'fleet_available', 'fleet_available_3',
                           'total_parcel_weight', 'avg_parcel_weight', 'avg_parcel_volume_size']]
        
        return self.date_df

    def save_results(self, filename: str):
        self.date_df.to_excel(f'{filename}.xlsx', index=False)