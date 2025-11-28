import pandas as pd

datasources = {"NQ", "ES"}

for datasource in datasources:
    # Read the CSV file
    df = pd.read_csv(f"{datasource}.csv")
    
    # Convert DateTime_ET to datetime and set as index
    df['DateTime_ET'] = pd.to_datetime(df['DateTime_ET'])
    df.set_index('DateTime_ET', inplace=True)
    
    # Resample to 5-minute intervals and aggregate OHLCV data
    # Open: first value in the period
    # High: maximum value in the period
    # Low: minimum value in the period
    # Close: last value in the period
    # Volume: sum of volumes in the period
    df_5min = df.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'DateTime_UTC': 'last'  # Keep the last UTC timestamp for reference
    })
    
    # Reset index to make DateTime_ET a column again
    df_5min.reset_index(inplace=True)

    # Save to new CSV file
    output_file = f"{datasource}_5min.csv"
    df_5min.to_csv(output_file, index=False)

    print(f"Converted {len(df)} 1-minute bars to {len(df_5min)} 5-minute bars")
    print(f"Output saved to: {output_file}")


