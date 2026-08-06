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
        
