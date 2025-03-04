import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
from collections import defaultdict

class StandardPrecipitationIndexValue:
    """Class to store SPI values."""
    def __init__(self, date, spi):
        self.date = date
        self.SPI = spi

class SPICollection:
    """Class to calculate SPI based on monthly precipitation data."""
    
    def __init__(self, time_scale):
        self.time_scale = time_scale
        self.SPI = []

    def aggregate_by_time_scale(self, precipitation):
        """Aggregate precipitation by time scale (monthly in this case)."""
        aggregated = defaultdict(float)
        for date, value in precipitation.items():
            key = (date.year, date.month)  # Group by Year-Month
            aggregated[key] += value
        return aggregated

    def calculate_by_month(self, precipitation):
        """Calculate SPI for each month across multiple years."""
        
        # Aggregate precipitation by month over all years
        aggregated_precipitation = self.aggregate_by_time_scale(precipitation)

        # Process each month separately (January to December)
        for month in range(1, 13):
            # Get precipitation values for the current month across years
            monthly_data = {date: value for date, value in aggregated_precipitation.items() if date[1] == month}

            if len(monthly_data) < 3:
                continue  # Not enough data for fitting
            
            values = np.array(list(monthly_data.values()))

            # Calculate SPI using gamma distribution
            shape, loc, scale = gamma.fit(values, floc=0)  # Force location to 0
            cdf = gamma.cdf(values, shape, loc=loc, scale=scale)
            spi_values = norm.ppf(cdf)

            # Save SPI values
            for idx, ((year, _), spi) in enumerate(zip(monthly_data.keys(), spi_values)):
                date = pd.Timestamp(year=year, month=month, day=1)
                self.SPI.append(StandardPrecipitationIndexValue(date, spi))

    def to_dataframe(self):
        """Convert SPI results to a Pandas DataFrame."""
        return pd.DataFrame([(spi.date, spi.SPI) for spi in self.SPI], columns=['Date', 'SPI'])

# Example Usage
if __name__ == "__main__":
    # Load precipitation data from CSV
    df = pd.read_csv("your_file.csv", parse_dates=['data'])
    df = df[['data', 'rrr24']].dropna()
    
    # Convert to dictionary {Date: Precipitation}
    precipitation_dict = dict(zip(df['data'], df['rrr24']))

    # Initialize and compute SPI
    spi_calculator = SPICollection(time_scale=1)
    spi_calculator.calculate_by_month(precipitation_dict)

    # Save results
    spi_df = spi_calculator.to_dataframe()
    spi_df.to_csv("spi_output.csv", index=False)

    print("SPI values calculated and saved to 'spi_output.csv'.")
