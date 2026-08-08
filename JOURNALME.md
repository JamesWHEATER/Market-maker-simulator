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

  08/08/2026 --
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




    
