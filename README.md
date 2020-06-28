# TechHype
Are daily stock price shifts more attributable to momentum or fundamentals?

Technology and technology-based companies have booming stock prices. Despite living in a bull market where the consistent yearly gain for the S&P 500 (an index tracking the 500 largest companies based on market cap) is 10%, technology stocks have exceeded that benchmark and returned well over other industries. We explore whether these exorbitant gains are a result of strong fundamentals, or if they are more attributable to hype and momentum in the market. In this study, we gather data surrounding the fundamentals behind a company and the hype or momentum that surrounds that company. From there, we compare each of these data points to movements of technology stock prices, and calculate correlations based on these factors. 

A natural question that results from this is how to quantify these companies’ fundamentals and their momentum. Most investors look at a few indicators for a company’s fundamentals, many of which we use as well: Price-to-Book (PB) Ratio and Price-to-Sales (PS) Ratio. These numbers are mostly ratios that reflect the strength of a company. For example, if a company has a high PB ratio compared to similar stocks, then its price is likely much higher compared to what it is earning. 

To measure hype, we look at the mentions that a particular company’s stock ticker (e.g. AAPL) has in Google searches. Additionally, we look at a stock’s Momentum Actual (MOA) and trading volume. MOA is a momentum indicator which measures the rate of change of a stock’s price--comparing its current value to its value from a few periods ago. Trading volume is a sum of how many shares of a stock were traded during a period of time.  These three data points combine to show a stock’s hype in the news and social media as well as how that hype is translated in the market both in terms of change in the stock’s price as well as change in its trading volume. 

Results

The percent change in stock price over a single day was modeled for five technology stocks: Netflix (NFLX), Roku (ROKU), Snap Inc. (SNAP), Tesla (TSLA), and Twitter (TWTR). For each stock, two linear regression models were learned, one using hype/momentum statistics ( weekly mentions, trading volume, and momentum) as explanatory variables, the other using financial fundamentals (price-to-book and price-to-sales ratios). Figure 1 shows the coefficients of multiple determination, or R2 values, of these two models for each stock. The coefficient of multiple determination is the percent of variance in the dependent variable, percent change in price, explained by the explanatory variables used in the model.



Figure 1: The R2 values for fundamental and hype models show the amount of variance in percent price change explained by the models.
The values shown in Figure 1 suggest that the hype model explains a great deal more variance than the fundamentals model does. On average for the five stocks used, the hype model explains 11.5 more times the variance in percent price change than the fundamentals model does.
An F-test was performed to test the validity of each model. The p-values for each model are shown in Table 1. At the significance level 0.05, the hype models are significant for every stock, but using fundamentals, only the model for Twitter is significant. The lack of statistical significance for four of the five fundamentals models suggests that the fundamental statistics used are not useful explanatory variables, and any explanation of variance from these models is likely due only to chance.

|               | Fundamentals | Hype |
| ------------- | ------------- | ------------- |
| NFLX  | 0.814  | 0.00121 |
| ROKU  | 0.191  | 0.00943 |
| SNAP  | 0.894  | 2.03E-08 |
| TSLA  | 0.346 | 4.60E-05 |
| TWTR  | 0.006 | 3.69E-05 |


