# from stock_predictor.stock_module.stock_screener import StockScreener

# stock_screener = StockScreener()
# filtered_tickers = stock_screener.filter_tickers()

# print(filtered_tickers)

################################################################

# from pprint import pprint

# import pandas as pd
# from stock_predictor.stock_module.stock_history import StockHistory

# stock_history = StockHistory()

# hisory_df = stock_history.daily_history("ERO")

# # print(hisory_df)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     pprint(hisory_df)

###############################################################
# from stock_predictor.stock_module.stock_technicals import StockTechnicals
# from stock_predictor.stock_module.stock_history import StockHistory

# history = StockHistory().daily_history("CX")

# stock_technicals = StockTechnicals().get_technicals_df("AAPL", history)

# print(stock_technicals)

###############################################################

# from stock_predictor.stock_module.stock_tickers import StockTickers
# from stock_predictor.stock_module.stock_industries import StockIndustries

# stock_industries = StockIndustries(0.4)

# industries = stock_industries.industry_avg_df()

# stock_tickers = StockTickers()
# tickers = stock_tickers.get_profitable_tickers(industry_avg_df=industries)

# print(tickers)
# print(len(tickers))

###############################################################
from stock_predictor.stock_module.stock_tickers import StockTickers

stock_tickers = StockTickers()

stock_tickers.new_add_tickers_to_db()

###############################################################
# from stock_predictor.stock_module.stock_history import StockHistory
# from stock_predictor.stock_module.stock_predictor import StockPredictor

# symbol = "AAPL"
# stock_history = StockHistory()
# stock_data = stock_history.daily_history(symbol)

# stock_predictor = StockPredictor(symbol, stock_data)
# predictions = stock_predictor.ExecuteModel()

# stock_predictor.plot_chart(predictions)
###############################################################

# from stock_predictor.stock_module.stock_data import StockData

# StockData().add_stored_tickers_to_db()
###############################################################

# from stock_predictor.stock_module.stock_history import StockHistory

# stock_history = StockHistory()

# hisory_df = stock_history.local_daily_history("WYNN")

# print(hisory_df)
