July --

  I built a random market generator where trades were 50/50 buy or sell, with volumes following a standard normal distribution. I also learned about market making and managing inventory risk. To top it off, I created a charting tool to visualize candlestick patterns from that market data.
  
04/08/2026 --

  In order to keep track of everything, I decided to journal my work. This will allow me to retrospectively view my findings and my labor. In order to do this, GPT told me that the best way was to document my work on Github. So, since I didn't know how GitHub worked, I spent the whole day figuring it out and learning how to use this powerful coding tool and linking everything I needed to my VS code.

  05/08/2026 -- 

  Today I decided, That's in order for me to continue my project, it would be wise to do some research on previous works about this subject so I learn and limits my potential mistakes. Today I read a work called: " What do we know about the profitability of technical analysis?".

  In this paper I learned what the skepticism towards technical analysis was:
  
  1.)  Many people accepted the efficient market hypothesis, which is hypothesis that explains that markets are efficient in information. Meaning, That's trying to exploits historic prices of a certain asset was futile since everyone has access to that kind of information.

  I learned that there Are three different types of efficiencies:

      1.) Weak efficiency: Were the information of the assets comes purely from the past price history, this is exploitable
      2.) Semi strong efficiency: where the information comes from all public available data, This includes Price history. This is very hard to exploit, Only data can exploit this type of markets
      3.) Strong efficiency: Where price reflects not only public but also private information, impossible to exploit
        
     2.) I also learned that people were skeptical about technical analysis because there were many negative empirical findings that couldn't prove that it actually worked in the stock market

I also learnt about a trade technical analysis trading strategy:

  I learned about the filter rules, which is a simple form of trading strategy, consisting of identifying trends based on percentage changes. A 1% filter rule, would decide if the price keeps going up, as soon as there is a 1% reversal going down, we would short the asset. The same is true for the bearish situation. The points of this trading strategy, is to try and catch the reversal wave and ride it. One would a tight stop loss, in this case usually of 1%, in case the reversal doesn't happen. Apparently, this trading strategy has the best performance between 1980 and 2000.

  I also learnt that There is a certain criteria to follow to be able to be considered as a viable and trustable research output, I was surprised to see, that many professional research papers didn't follow all of these criteria. Which is Y I will try to follow all of them in my research. The criteria consists of the following:

   1.) Transaction costs, where we should take into account not only brokerage fees but also bid and ask spread
   2.) Risk adjustment
   3.) Trading rule optimization, which is when you try to find the best trading strategy that fits your data
   4.) Out of sample test
   5.) Statistical tests
   6.) Data snooping addressed


  It was interesting to see that the only post work that actually satisfy these six criteria, was a genetic programming study and in this study they tried to beat the S and P500. However it suggested that no matter what they did, they could not beat it. The paper, repeatedly explains that's trying to use technical analysis to trade stocks or indexes of stocks such as the S and P 500 was not going to work because these markets are by far the most efficient and therefore disable technical analysis traders to be profitable. The paper gives us the following two reasons:

 1.) Many studies, in all the and modern times, when testing specifically on these types of markets, they cannot be profitable
 2.) This is because of the rise of liquidity high competition which stops lag between information and prices on this market. Indeed, It is very hard nowadays be the first person to view a specific news on the stock such as quarterly returns since this type of information is public and can be viewed almost immediately when it's out.

  Therefore, the paper suggests that's technical analysis trading is much more effective on other types of markets such as future markets and foreign exchange markets. So in the future, if I will trade using technical analysis strategies, I will trade on those markets.


  The paper also mentions chart pattern studies, which is more relevant to this project. It was very interesting to see that there were mixed results, Two previous studies about this found positive results, two previous studies about this found negative results, and one previous study found mix results. It's very interesting to note that all of these studies had data snooping problems, so in my study I want to make sure I don't have this problem, first idea that I have is I can generate a lot a lot of different seeds, to be sure I am not just getting lucky on some of the seeds I can generate as many seeds as I want.

  Finally, The paper comes up with explanations for technical trading profits:
  
  1.) Noisy information: In many markets prices adjust to information, but they do so slowly, This lag is exploitable. For example, central bank interventions Could be considered as the exploitable information. You just have to think deeper than the other people, of course the direct implications of a central bank intervention on the will not be exploitable because too many people will try to exploit them, however the indirect implications can still be exploitable.
    
  2.) Behavioral models: Although assets have their unique price, in reality, The price is controlled by supply and demand, but this supply and demand don't use the same strategy. Some of these traders are behavioral traders, Where they don't actually try to predict the price, they just try to speculate, If too many of these traders arrive, they can Misprice the assets based on their speculation.
    
  3.) Herding model: This is the same concept as FOMO, when price gets caught in a trend, many FOMO traders will not try to understand why the price is moving and make their own prediction, and instead try to catch on the trend to not miss out on potential gains. This creates a surplus in demand for supply, Which makes it exploitable for technical analysis traders.
    
  4.) Chaos theory: This is just a fancy way of saying traders get lucky. Since real life markets are extremely complex, Most of the time, The exact reason why a person would lose or win a trade is unknown, and if a tiny detail would have changed in the past, That could have moved the market a lot and changed the results of that trade astronomically.
    
  5.) Order flow: Order flow is very important in technical analysis trading, since many people decide to enter a position on round numbers ending with zeros, This could be because typing these types of numbers reduce typing errors, could also be because having to mentally remember or use these numbers in mathematical calculations is easier, could also be just because people prefer this type of numbers. So, since many people enter their positions in round numbers prices, such as 2000, we can clearly see in the order book a clustering of positions around round numbers, This makes support and resistance particularly a true concept and is exploitable by technical analysis traders.
    
  6.) Temporary market inefficiencies: This basically a concept saying that the more people know about your specific working strategy the less it actually works. This is because if there is an exploitable hole in the market, then many people will start using it, which will force the markets to react to it, and make the strategy not profitable anymore. Which is the main reason why large banks do not want to share their profitable strategies with everyone, because as soon as they share it their strategy isn't profitable anymore.
    
  7.) Risk premiums: Many time we can see on the Internet people win an Astronomically huge win while trading. However we need to take consider into account their risk premiums, once their risk is adjusted in our calculations, only can we compare to other trades
    
  8.) Transaction costs: Of course, having lower transaction costs such as brokerage commissions and bid ask spread will make you more profitable

  All of this, gives me the idea of trying to find a market with as many exploitable holes in it as possible in the future. This research paper truly give me some good ideas of what to do on this project, however I still want to read one more paper about something more specific to my projects, specifically about pattern training.

  07/08/2026 --

today I read two papers. the first one, entitled " what makes trading strategies based on chart pattern recognition profitable" by Prodromos Tsinaslanidis and Francisco Guijarro. this paper was very interesting, but it wasn't really relevant to my project. The paper analy different types of chat patterns than the one I want to analyse in my project. I want to analyse commonly used patterns by retail traders, such as head and shoulders and flags. However this paper uses original approach of what the pattern is to be considered They use something called DTW (Dynamic time warping). this trading tool tries to scan a specific pattern in one market and see if it aligns in historical data of different markets, the word warping is used here because the tool stretches and manipulates the charts in order to find similar chart patterns as current asset. in this paper they use this tool coupled with a tool called UCR(University of California Riverside suite) which is a tool that allows the researcher to use DTW on a very large data set and analyse the data very quickly and efficiently.

it was very interesting to see that's the most optimal configurations of this trading strategy were:
  1.) the of the chart pattern used should be either 15, 20, Or 25. This means that the algorithm should use either the 15, 20, or 25th most recent candles to analyse that whole chart pattern across markets
  2.) the number of references should be either 10, 15, or 20 . This is tell the algorithm How many similar chart patterns to the current markets it should find across different markets
  3.) as a general rule, the stop loss should be under or equal to the take profit. This will make the winner trades appear more strongly as the end result
  4.) finally the consensus, which is a percentage that measures how many actual trades we go in out of the number of all possible trades. This number should be between 0.5% and 7.5%. so this consensus basically is measured about how we tell the programme to act. So if there are 10 reference charts, we should tell the algorithm to only enter the position For example a long position if or nine out of those 10 references are also bullish when this chart pattern comes up.

  in this paper, the number of positive trading systems According to these measures Where was 92.5% of all trading systems. The average return after transactions was 0.13% a day, which is equal to an annual expected return of 60%, which is really good.

  this research paper, Did all of those experiments on New York Stock Exchange stocks, which as we know thanks to the lost research paper, are the most efficient markets available. so even though in this markets technical traders are at a disadvantage, The expected annual return is still 60%, which is very interesting. In the future I would like to use this type of trading technique on a less efficient market such as futures or exchange rates markets, Using best configuration such as high take profits and query lengths plus lower number of references and lower stop losses. Since lower number of references will make the tools only selected the most appropriate similar charts.

  08/08/2026 + 09/08/2026 --
    I have started to read a new paper entitled:" foundations of technical analysis; Computational algorithms; Statistical inference, an empirical implementation" by ANDREW W. LO, HARRY MAMAYSKY, AND JIANG WANG.

    it is a very interesting paper that is very aligned with my project, from what I understood so far, It is teaching me how help an algorithm mathematically recognise different trading patterns.

    I have only read the first parts for now, which sets the base for the mathematical foundations and techniques in the paper.

    First, I learned what's in Monte Carlo simulation is. It is a way to analyse if a certain given result comes from a randomness or if it is genuinely related to a problem. and for more I understood, the goal is to generate a tonne of fake data under a random model, and then compare it to real results to see if the random data was actually predictive and works. This is very similar to what I'm doing, So maybe I will switch my simulation to a Monte Carlo simulation. by law of large numbers, the prediction of a Montecolor simulation should approximate the true probability, if the data set is large enough.

    in the paper, We assume that prices follow the following equation: Pt =m(Xt) + DELTAt, where Pt is the assets price at a given time t and,  m(Xt) is the true nonlinear function of price and DELTAt is noise at time t.

    secondly, in order to approximate the true nonlinear function m(x), we use something called a smoothing estimator. the goal, is to get the weighted average between the price of an asset at a given time T and the weight. where prices are multiplied by their weight, their weights being the density of the price in our given sample.

    furthermore, in order to get this density function, we use something called a kernel regression. A kernel regression is a technique used to approximate true population density from small data samples. The idea is, each data in our sample is gonna have its own kernel function, in our case and the most popular case, The kernel function would be the standard distribution of the data point. then, we use each data points kernel function to try to approximate the density of nearby possible data points, we would give more weight to data points that are closer to the specific data point that we're trying to calculate. For example, If we are trying to find out what is the approximate density of students getting the score of 50 out of 100 on the test, the kernel functions of the scores 49 and 51 would have much more weight than the kernel functions of scores one and 99.

    another thing to consider, is the bandwidth of each of these kernel function. If the bandwidth it's too large, this would give us an over smoothed density curve that would suggest a uniform distribution, which would be useless for our continuing calculations. Similarly, a too little bandwidth, would give us a hyper narrow under smooth curve, That would be too sensitive to our data points in our actual data sets and would be useless. therefore, it is important for us to get the proper bandwidth to get that sweet spot.

    the kernel density function is given by the following: <img width="586" height="129" alt="image" src="https://github.com/user-attachments/assets/0f0e664e-eabd-438b-98bf-f4b631bac9e9" />
where H is the variable that controls the bandwidth of the kernel function. The Bigger the H, the points that were previously far away from our price are now closer to us. Which means that they will have a higher weight that takes into consideration in our kernel function. Which also means that they will have a higher density. And therefore a higher H will produce a wider bandwidth.

  now that we can have density function, we can now provide with an equation that approximates the nonlinear price of our asset at any point:<img width="849" height="858" alt="image" src="https://github.com/user-attachments/assets/8422425a-74f4-452a-ac40-da16674a6bd0" />

  finally call mom It is also crucial to be able to select the optimal H to set the optimal bandwidth of the kernel functions. To do this we use something called a cross validation where H is chosen to minimise the following equation: <img width="645" height="203" alt="image" src="https://github.com/user-attachments/assets/bdfc9227-5d24-447c-9dbb-6b8262b78398" />. basically, the idea is to approximate the nonlinear function M without using a specific data point and then tested the residual squares, which is basically the average of the squared difference between the true asset price- The estimated nonlinear asset price.
  interestingly, the optimal H generating by this technique over smoothed the kernel function when it came to the technical analysis scenario in the paper. As a solution The paper continued to use a new optimal solution which was 30% of the initial calculated H. This is not the most rigorous way to find the optimal, and leads us to question why this method didn't work for our project.

  now that we got the math out the way, our next step is to construct the detection of technical patterns algorithm. For this the paper suggests three steps:
    1.) define each technical pattern in terms of geometric properties, for example, local extrema (maximum and minima)
    2.) construct a kernel estimator M of a given time series of prices so that its extra mark can be determined numerically
    3.) analyse M for occurrences of each technical pattern

so an example of mathematically defining chart patterns, for a head and shoulders pattern, You would E1 E3 E5 as local maxima points, and E-2 with E4 local minima. Then you will say that E3 has to be bigger than E1 And that E3 also has to be bigger than E5. so here E3 would be your head, E1 and E5 would be your shoulders. Then you should say that E1 and E5 are within 1.5% of their average, because the shoulders need to be aligned. E-2 would be the beginning or end of the shoulders since they would represent the local minima, they too have to be within 15 percent of their average.

  then we have to use a window, to help the algorithm focus on the most recent pattern making, and not take all the data into consideration in one go, because if it does that there will be too much noise and impossible for it to find out if the patterns have predictable abilities. We usually do 35 day window, and then the paper added a an extra three days as a kind of buffer for the programme to be able to catch and finalise its patterns.

  in the paper to find the local extremes, we would compare the signs of the derivative of M at the time T and T plus 1. so if they have different signs of course we would have passed an extrema. Furthermore, If we we find a time that has a derivative of 0, We still need analyse if it is an extrema, which is why we introduce a new variable and calculate the sine of the derivative of of T1 prove the sign of S which is just the next time frame after T that has the derivative of M not equal to zero.

  finally, After getting the pattern results, we should use goodness of fits test or another test to if the patterns truly have their own predictability ability. So the idea, is to compare the distribution of the conditional probability which is when we have used the patterns with the unconditional probability which is just normal unconditional Probability of the market. And if it is different, then patterns have their own predictability ability. The paper also wanted to to assume that volume also has its own predictability ability so it also tested the volume.

  as a conclusion of the paper, I found that's seven out of 10 of the patterns that we tested had a different distribution to the normal unconditional Distribution. All of them gave a very light statistical edge of the market. However, it didn't analyse if this statistical edge was exploitable. I might be able to prove if it is exploitable with my own project. And as a hunch, This very light statistical edge is probably not exploitable after transaction costs, since these types of minute advantages are often illuminated after transaction costs are taken into consideration.
10/08/2026 --

  Today I started moving from the theory in the Lo, Mamaysky and Wang paper into my own actual detector. The main thing I wanted was to stop chart patterns from being something visual and subjective and turn them into rules that a computer can apply exactly the same way every time.

  I worked on defining the detector as a separate module from the simulator and chart renderer. This is important because I want the same detector to work on synthetic markets and later on real BTC/ETH data without changing the rules.

  I also focused on the idea of look-ahead bias. A pattern should only be considered available after the information needed to confirm it has actually happened. The detector should never be allowed to look at the future and then pretend that it knew the pattern earlier.

12/08/2026 --

  I continued developing the classical pattern detector. I worked on smoothing the price series, detecting local extrema and then applying geometric rules to those extrema.

  I decided that I wanted the detector to be a non-AI baseline. The reason is that if I use AI from the beginning, I would never know if the AI was actually adding value or just learning the pattern definitions itself.

  The detector therefore has to answer only:

  "Was a valid pattern present here?"

  Profitability is kept completely separate and only measured afterwards.

14/08/2026 --

  Today I worked on making the pattern detector more research-safe and reproducible.

  I added the idea that each detection should store:
  - pattern type
  - start and end candle
  - direction expected by the classical interpretation
  - geometry fit score
  - the time when the pattern became available
  - the earliest candle where a trade is actually allowed to execute
  - metadata describing the shape

  One thing I found particularly important is separating "available_at_index" from "earliest_execution_index". If I detect something using the close of candle t, I should not then allow myself to trade at that same already-known close. The first possible action has to come afterwards.

17/08/2026 --

  I spent a lot of time auditing the detector and the logic around it. I did not want to simply copy Lo's algorithm line by line. I wanted a detector that is strongly inspired by the same scientific ideas but is also appropriate for my own synthetic markets and later real data.

  I worked on making the geometry rules scale-free so that a pattern definition does not depend on whether an asset trades at 100, 1,000 or 100,000.

  I also started expanding the detector beyond only one pattern. The architecture was designed so that patterns such as head and shoulders, inverse head and shoulders, double tops, double bottoms, triangles, rectangles and broadening formations could all be detected in the same framework.

18/08/2026 --

  Today I worked heavily on the synthetic market side of the project.

  The simulator evolved from a simple random market into a market with nine possible trader/mechanism worlds:

  1.) random
  2.) rule-based
  3.) emotional
  4.) information
  5.) mean-reversion
  6.) momentum
  7.) regime
  8.) liquidity
  9.) adaptive

  These worlds can be used individually or mixed together.

  This is a major step for the project because instead of only asking whether a pattern worked on one random chart, I can now ask what underlying market structure may have caused it to work.

  I also made the simulator more realistic by having independent trader arrivals, bid/ask quotes, market-maker inventory, spread effects and different types of order flow rather than scripting price paths directly.

20/08/2026 --

  Today I connected the whole classical research stack more seriously:

  simulator -> candles -> pattern detector -> backtester/statistics.

  I wanted every component to stay modular so that later I can replace synthetic candles with real-market candles without changing the detector.

  I also worked on the market-structure experiment. The idea is to test every non-empty combination of the nine worlds. Since there are 9 possible worlds, there are 2^9 - 1 = 511 non-empty combinations.

  The experiment is not supposed to simply find the highest-return structure and stop there. I added safeguards such as:
  - repeated seeds
  - transaction costs
  - matched null comparisons
  - out-of-sample validation
  - multiple-testing controls
  - no retuning the detector based on realised profits

  This is directly related to the data-snooping problems I read about earlier.

21/08/2026 --

  Today I spent time on real-market backtesting.

  I used frozen Binance BTC and ETH historical data and tested the same detector/backtesting logic across several candle horizons.

  One important thing I clarified for myself is the meaning of the 20-bar horizon. It means that once the trade is entered, the backtest exits 20 candles later. There is no stop-loss or take-profit in this baseline test. I wanted a simple fixed-horizon measurement first so that results are directly comparable.

  Transaction costs include both brokerage and bid/ask spread.

  The early real-market results showed that the very short timeframes were generally bad after costs, while some longer horizons looked more promising.

23/08/2026 --

  Today I focused on preparing the project for the Polymer Technology Exposition application.

  I created a one-page project write-up around the research question:

  "Under what market conditions does pattern trading have a predictable edge?"

  I had to explain the project very concisely:
  - problem statement
  - simulator
  - pattern detector
  - market-structure experiment
  - AI layer
  - real-market backtesting
  - impact
  - reflections

  I also prepared the video submission. This forced me to understand how to explain the technical parts simply instead of just being able to code them.

25/08/2026 --

  I continued refining the experiment logic after the Polymer submission.

  One important result from the synthetic market-structure experiment was the structure that performed best in the earlier screen:

  60% rule-based
  20% emotional
  20% mean-reversion

  The intuition I developed is that rule-based traders can extend moves, emotional traders can create overshoots, and mean-reversion traders can pull price back. This push-pull interaction can naturally create retests and shapes that resemble shoulders, double tops/bottoms and triangles.

  This was interesting because it gave a possible mechanism for why chart patterns could become informative in some markets but not others.

27/08/2026 --

  I worked on improving the real-market comparison.

  The most interesting real-market results were not on the shortest candles.

  The results I had at this point included:
  - 1m and 15m generally negative after costs
  - 30m mostly negative
  - 1h 20-bar all-pattern result: BTC about +0.163% average net/trade and ETH about +0.118%
  - daily 20-bar results were much stronger but with fewer observations
  - weekly data was too sparse to make a reliable conclusion

  I also found a particularly strong BTC daily Double Bottom result at the 20-bar horizon, but the number of trades was small, so I treated it as something interesting to validate rather than proof of a robust strategy.

29/08/2026 --

  Today I moved more seriously into the AI part of the project.

  I decided the cleanest design was not to ask AI to invent patterns from raw charts. Instead:

  1.) the classical detector identifies a pattern
  2.) that pattern occurrence becomes one observation
  3.) the AI sees information that was available at the time
  4.) future prices are used only afterwards to label whether that occurrence succeeded after costs

  This means the AI research question is:

  "Given that a pattern exists, what market conditions make this occurrence more likely to succeed?"

  This seems much more aligned with the overall research question than simply training a black-box model to predict prices.

31/08/2026 --

  I worked on the AI feature design.

  I separated features into two groups:

  Observable:
  information that could realistically be available in a real market, such as:
  - pattern type
  - pattern geometry
  - recent return
  - trend
  - volatility
  - range
  - volume
  - order-flow imbalance
  - order-flow persistence
  - spread
  - liquidity / price-impact proxies

  Oracle:
  all Observable information plus hidden simulator information such as:
  - true world weights
  - emotional FOMO/fear parameters
  - informed trader fraction
  - momentum strength
  - mean-reversion strength
  - regime persistence
  - liquidity state
  - synthetic sentiment and information state

  The Observable model is the bridge to real markets. The Oracle model is mainly a research tool to help explain which hidden mechanisms may be associated with pattern profitability.

02/09/2026 --

  Today I worked on data leakage and the train/validation/test structure.

  Instead of randomly splitting individual pattern rows, I split entire synthetic market structures.

  The split is:

  60% training
  20% validation
  20% untouched testing

  This is important because if patterns from the same synthetic structure appeared in both training and testing, the model could learn structure-specific behaviour and make the test results look better than they really are.

  The validation set is used to select the AI probability threshold. The test set is left untouched until the model and threshold have already been chosen.

04/09/2026 --

  I continued building and auditing the AI pipeline.

  The models I kept were:
  - Logistic Regression as a simple interpretable baseline
  - XGBoost for nonlinear relationships
  - MLP as a neural-network comparison

  XGBoost is also useful because I can export feature importance and SHAP summaries to understand which conditions are associated with model predictions.

  I also made sure that future/outcome columns such as profitability, future return, entry/exit results and favourable/adverse excursion cannot accidentally appear in the model input features.

  This is one of the most important safeguards in the entire AI stage because future prices are only allowed to create the label, never the inputs.

07/09/2026 --

  Today I did a very detailed audit of how all the files work together.

  I checked the interaction between:
  - market_maker_simulator.py
  - chart_renderer.py
  - pattern_detector.py
  - market_structure_experiment.py
  - ai_pattern_research_pipeline.py

  The reason was simple: even if each file works individually, the research results would still be wrong if they disagree about candle construction, execution timing, costs, pattern events or data provenance.

  I paid particular attention to:
  - look-ahead bias
  - earliest execution timing
  - whether raw candidates were accidentally counted as independent patterns
  - transaction-cost consistency
  - structure leakage between AI splits
  - deterministic seeds and reproducibility
  - source/config fingerprints

  The code became much closer to a proper research pipeline rather than a collection of separate scripts.

08/09/2026 --

  Today I focused on how the AI dataset should actually be generated and trained at scale.

  I clarified the difference between "structures" and "seeds".

  A structure is one set of market-generating conditions. A seed is one random realisation of that same structure.

  So:

  Structure A, seed 1
  Structure A, seed 2
  Structure A, seed 3

  are not three different market structures. They are repeated random paths under the same underlying market conditions.

  This is useful because I want the model to learn whether a market mechanism is robust rather than learning one lucky path.

  I also clarified that the simulator's own Monte Carlo mode is different. That mode simply repeats the market and averages simulator statistics. It does not run the pattern detector. Pattern-level Monte Carlo / repeated-seed research happens in the experiment and AI pipelines.

09/09/2026 --

  Today I prepared the project for the actual Polymer Technology Exposition tomorrow.

  I organised the material I want to bring:
  - submitted one-page project overview
  - source code
  - research papers
  - journal
  - terminal command guide
  - synthetic market charts
  - detector outputs
  - AI results
  - real-market backtesting results

  I decided that the submitted PDF already works as a good high-level architecture overview, so I do not need to waste time showing a second almost-identical architecture slide unless somebody asks about the technical module structure.

  I also prepared a short live demo.

  The intended demo is:

  1.) generate a synthetic market using the winning 60/20/20 structure
  2.) open the generated OHLCV candlestick chart
  3.) run the classical detector on that exact generated candle data
  4.) show one detected Head & Shoulders occurrence
  5.) explain when the pattern became available and when execution is first allowed
  6.) show that future prices are only examined after detection
  7.) connect that occurrence to the AI pipeline
  8.) show how the AI tries to learn which market conditions make detected patterns worth trusting

  I also searched through many deterministic seeds to find a presentation-friendly Head & Shoulders example in the winning market structure.

  Seed 41 was a particularly good example. The detector found a clear H&S with:
  - left shoulder around 102.83
  - head around 103.48
  - right shoulder around 102.90
  - almost-flat neckline troughs around 99.76 and 99.89

  The pattern occurs around candles 416-462, becomes available later, and the earliest execution is after that information point. The price subsequently moves lower, which makes it a useful visual demonstration of the complete idea.

  I also created a terminal command reference for the project so I can quickly remember how to:
  - generate different market worlds
  - change weights, seeds and market settings
  - use simulator Monte Carlo mode
  - run the detector
  - build the AI dataset
  - train Observable / Oracle models

  Looking back at the project, the biggest change is that it started as a simple random market generator and became a full research framework.

  The progression was roughly:

  random market
  -> market maker
  -> nine synthetic trader worlds
  -> OHLCV candle generation
  -> classical pattern detector
  -> causal execution/backtesting
  -> 511 market-structure experiment
  -> real BTC/ETH validation
  -> leakage-safe AI dataset
  -> Observable / Oracle models
  -> explainability and out-of-sample testing

  The most important lesson so far is that finding a pattern is easy compared with proving that it contains a genuine economic edge.

  Every time the project became more rigorous, I found another way that a result could accidentally look better than reality: transaction costs, look-ahead bias, data snooping, repeated observations from the same structure, multiple testing, lucky seeds, or future information leakage.

  That has probably been the most useful part of the whole project. I did not just learn how to build a trading algorithm. I learned how difficult it is to design an experiment that I can actually trust.

  Tomorrow, my goal is not to claim that chart patterns have been "proven" profitable. My goal is to show the research system I built to answer a much narrower and more interesting question:

  "Under what market conditions does a detected chart pattern have a predictable edge?"
