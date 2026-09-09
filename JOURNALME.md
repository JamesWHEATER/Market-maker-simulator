July --

I built a random market generator where trades were 50/50 buy or sell, with volumes following a standard normal distribution. I also learned about market making and managing inventory risk. To top it off, I created a charting tool to visualize candlestick patterns from that market data.

04/08/2026 --

In order to keep track of everything, I decided to journal my work. This will allow me to retrospectively view my findings and my labor. In order to do this, GPT told me that the best way was to document my work on Github. So, since I didn't know how GitHub worked, I spent the whole day figuring it out and learning how to use this powerful coding tool and linking everything I needed to my VS code.

05/08/2026 --

Today I decided, That's in order for me to continue my project, it would be wise to do some research on previous works about this subject so I learn and limits my potential mistakes. Today I read a work called: " What do we know about the profitability of technical analysis?".

In this paper I learned what the skepticism towards technical analysis was:

1.) Many people accepted the efficient market hypothesis, which is hypothesis that explains that markets are efficient in information. Meaning, That's trying to exploits historic prices of a certain asset was futile since everyone has access to that kind of information.

I learned that there Are three different types of efficiencies:

  1.) Weak efficiency: Were the information of the assets comes purely from the past price history, this is exploitable
  2.) Semi strong efficiency: where the information comes from all public available data, This includes Price history. This is very hard to exploit, Only data can exploit this type of markets
  3.) Strong efficiency: Where price reflects not only public but also private information, impossible to exploit
    
 2.) I also learned that people were skeptical about technical analysis because there were many negative empirical findings that couldn't prove that it actually worked in the stock market
I also learnt about a trade technical analysis trading strategy:

I learned about the filter rules, which is a simple form of trading strategy, consisting of identifying trends based on percentage changes. A 1% filter rule, would decide if the price keeps going up, as soon as there is a 1% reversal going down, we would short the asset. The same is true for the bearish situation. The points of this trading strategy, is to try and catch the reversal wave and ride it. One would a tight stop loss, in this case usually of 1%, in case the reversal doesn't happen. Apparently, this trading strategy has the best performance between 1980 and 2000.

I also learnt that There is a certain criteria to follow to be able to be considered as a viable and trustable research output, I was surprised to see, that many professional research papers didn't follow all of these criteria. Which is Y I will try to follow all of them in my research. The criteria consists of the following:

1.) Transaction costs, where we should take into account not only brokerage fees but also bid and ask spread 2.) Risk adjustment 3.) Trading rule optimization, which is when you try to find the best trading strategy that fits your data 4.) Out of sample test 5.) Statistical tests 6.) Data snooping addressed

It was interesting to see that the only post work that actually satisfy these six criteria, was a genetic programming study and in this study they tried to beat the S and P500. However it suggested that no matter what they did, they could not beat it. The paper, repeatedly explains that's trying to use technical analysis to trade stocks or indexes of stocks such as the S and P 500 was not going to work because these markets are by far the most efficient and therefore disable technical analysis traders to be profitable. The paper gives us the following two reasons:

1.) Many studies, in all the and modern times, when testing specifically on these types of markets, they cannot be profitable 2.) This is because of the rise of liquidity high competition which stops lag between information and prices on this market. Indeed, It is very hard nowadays be the first person to view a specific news on the stock such as quarterly returns since this type of information is public and can be viewed almost immediately when it's out.

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

it was very interesting to see that's the most optimal configurations of this trading strategy were: 1.) the of the chart pattern used should be either 15, 20, Or 25. This means that the algorithm should use either the 15, 20, or 25th most recent candles to analyse that whole chart pattern across markets 2.) the number of references should be either 10, 15, or 20 . This is tell the algorithm How many similar chart patterns to the current markets it should find across different markets 3.) as a general rule, the stop loss should be under or equal to the take profit. This will make the winner trades appear more strongly as the end result 4.) finally the consensus, which is a percentage that measures how many actual trades we go in out of the number of all possible trades. This number should be between 0.5% and 7.5%. so this consensus basically is measured about how we tell the programme to act. So if there are 10 reference charts, we should tell the algorithm to only enter the position For example a long position if or nine out of those 10 references are also bullish when this chart pattern comes up.

in this paper, the number of positive trading systems According to these measures Where was 92.5% of all trading systems. The average return after transactions was 0.13% a day, which is equal to an annual expected return of 60%, which is really good.

this research paper, Did all of those experiments on New York Stock Exchange stocks, which as we know thanks to the lost research paper, are the most efficient markets available. so even though in this markets technical traders are at a disadvantage, The expected annual return is still 60%, which is very interesting. In the future I would like to use this type of trading technique on a less efficient market such as futures or exchange rates markets, Using best configuration such as high take profits and query lengths plus lower number of references and lower stop losses. Since lower number of references will make the tools only selected the most appropriate similar charts.

08/08/2026 + 09/08/2026 -- I have started to read a new paper entitled:" foundations of technical analysis; Computational algorithms; Statistical inference, an empirical implementation" by ANDREW W. LO, HARRY MAMAYSKY, AND JIANG WANG.

it is a very interesting paper that is very aligned with my project, from what I understood so far, It is teaching me how help an algorithm mathematically recognise different trading patterns.

I have only read the first parts for now, which sets the base for the mathematical foundations and techniques in the paper.

First, I learned what's in Monte Carlo simulation is. It is a way to analyse if a certain given result comes from a randomness or if it is genuinely related to a problem. and for more I understood, the goal is to generate a tonne of fake data under a random model, and then compare it to real results to see if the random data was actually predictive and works. This is very similar to what I'm doing, So maybe I will switch my simulation to a Monte Carlo simulation. by law of large numbers, the prediction of a Montecolor simulation should approximate the true probability, if the data set is large enough.

in the paper, We assume that prices follow the following equation: Pt =m(Xt) + DELTAt, where Pt is the assets price at a given time t and,  m(Xt) is the true nonlinear function of price and DELTAt is noise at time t.

secondly, in order to approximate the true nonlinear function m(x), we use something called a smoothing estimator. the goal, is to get the weighted average between the price of an asset at a given time T and the weight. where prices are multiplied by their weight, their weights being the density of the price in our given sample.

furthermore, in order to get this density function, we use something called a kernel regression. A kernel regression is a technique used to approximate true population density from small data samples. The idea is, each data in our sample is gonna have its own kernel function, in our case and the most popular case, The kernel function would be the standard distribution of the data point. then, we use each data points kernel function to try to approximate the density of nearby possible data points, we would give more weight to data points that are closer to the specific data point that we're trying to calculate. For example, If we are trying to find out what is the approximate density of students getting the score of 50 out of 100 on the test, the kernel functions of the scores 49 and 51 would have much more weight than the kernel functions of scores one and 99.

another thing to consider, is the bandwidth of each of these kernel function. If the bandwidth it's too large, this would give us an over smoothed density curve that would suggest a uniform distribution, which would be useless for our continuing calculations. Similarly, a too little bandwidth, would give us a hyper narrow under smooth curve, That would be too sensitive to our data points in our actual data sets and would be useless. therefore, it is important for us to get the proper bandwidth to get that sweet spot.

the kernel density function is given by the following: <img width="586" height="129" alt="image" src="https://github.com/user-attachments/assets/0f0e664e-eabd-438b-98bf-f4b631bac9e9" />
where H is the variable that controls the bandwidth of the kernel function. The Bigger the H, the points that were previously far away from our price are now closer to us. Which means that they will have a higher weight that takes into consideration in our kernel function. Which also means that they will have a higher density. And therefore a higher H will produce a wider bandwidth.

now that we can have density function, we can now provide with an equation that approximates the nonlinear price of our asset at any point:image

finally call mom It is also crucial to be able to select the optimal H to set the optimal bandwidth of the kernel functions. To do this we use something called a cross validation where H is chosen to minimise the following equation: image. basically, the idea is to approximate the nonlinear function M without using a specific data point and then tested the residual squares, which is basically the average of the squared difference between the true asset price- The estimated nonlinear asset price. interestingly, the optimal H generating by this technique over smoothed the kernel function when it came to the technical analysis scenario in the paper. As a solution The paper continued to use a new optimal solution which was 30% of the initial calculated H. This is not the most rigorous way to find the optimal, and leads us to question why this method didn't work for our project.

now that we got the math out the way, our next step is to construct the detection of technical patterns algorithm. For this the paper suggests three steps: 1.) define each technical pattern in terms of geometric properties, for example, local extrema (maximum and minima) 2.) construct a kernel estimator M of a given time series of prices so that its extra mark can be determined numerically 3.) analyse M for occurrences of each technical pattern

so an example of mathematically defining chart patterns, for a head and shoulders pattern, You would E1 E3 E5 as local maxima points, and E-2 with E4 local minima. Then you will say that E3 has to be bigger than E1 And that E3 also has to be bigger than E5. so here E3 would be your head, E1 and E5 would be your shoulders. Then you should say that E1 and E5 are within 1.5% of their average, because the shoulders need to be aligned. E-2 would be the beginning or end of the shoulders since they would represent the local minima, they too have to be within 15 percent of their average.

then we have to use a window, to help the algorithm focus on the most recent pattern making, and not take all the data into consideration in one go, because if it does that there will be too much noise and impossible for it to find out if the patterns have predictable abilities. We usually do 35 day window, and then the paper added a an extra three days as a kind of buffer for the programme to be able to catch and finalise its patterns.

in the paper to find the local extremes, we would compare the signs of the derivative of M at the time T and T plus 1. so if they have different signs of course we would have passed an extrema. Furthermore, If we we find a time that has a derivative of 0, We still need analyse if it is an extrema, which is why we introduce a new variable and calculate the sine of the derivative of of T1 prove the sign of S which is just the next time frame after T that has the derivative of M not equal to zero.

finally, After getting the pattern results, we should use goodness of fits test or another test to if the patterns truly have their own predictability ability. So the idea, is to compare the distribution of the conditional probability which is when we have used the patterns with the unconditional probability which is just normal unconditional Probability of the market. And if it is different, then patterns have their own predictability ability. The paper also wanted to to assume that volume also has its own predictability ability so it also tested the volume.

as a conclusion of the paper, I found that's seven out of 10 of the patterns that we tested had a different distribution to the normal unconditional Distribution. All of them gave a very light statistical edge of the market. However, it didn't analyse if this statistical edge was exploitable. I might be able to prove if it is exploitable with my own project. And as a hunch, This very light statistical edge is probably not exploitable after transaction costs, since these types of minute advantages are often illuminated after transaction costs are taken into consideration.

10/08/2026 – 18/08/2026
This week was probably the most important week of the project so far because I moved from mainly reading about how technical-pattern research works to actually trying to build the complete research system myself.

Until now, I had a general idea of what I wanted to investigate: whether commonly recognised chart patterns such as Head and Shoulders, Double Tops, Double Bottoms, triangles, and other formations actually contain information about future prices. However, this week made me realise that there is a huge difference between having that research question and having a program that can answer it scientifically.

The project is no longer simply "make some artificial prices and see whether patterns work." I am beginning to construct something much closer to an experimental framework.

The general architecture of my project has now become:

Synthetic market → price and volume data → pattern detector → pattern classification → future-return measurement → statistical analysis → comparison with real financial markets.

One of the most important things I learned this week is that every one of these stages can introduce its own biases. Therefore, building a good research project is not only about writing an algorithm that works. It is about making sure that the information given to the algorithm, the assumptions used in the simulation, and the way the results are evaluated do not accidentally create the result I am looking for.

Developing the synthetic markets
At the beginning of the project, my synthetic market was extremely simple. I had essentially created a random market where buys and sells occurred with approximately equal probability and trade volumes were randomly generated.

That was useful because it gave me a baseline. If my pattern detector finds extremely profitable patterns in a completely random market, that would immediately suggest that something is wrong with either my detector, my testing method, or the statistical interpretation of the results.

However, I realised that a purely random market is not enough.

Real financial markets contain many mechanisms which can create temporary structure in prices: trends, mean reversion, differences in liquidity, order-flow imbalance, different trader behaviours, volatility changes, market-maker inventory management, and many other effects.

Therefore, I spent a significant amount of time this week improving the market-maker simulator.

One of the biggest conceptual lessons was understanding that the market maker should not simply generate prices randomly. A market maker continuously reacts to incoming buy and sell orders while simultaneously managing its own inventory.

For example, if the market maker has accumulated too much inventory because many traders have been selling to it, continuing to quote the same prices would expose the market maker to increasing risk. It therefore has an incentive to change its quotes in a way that encourages the market to buy some inventory back.

This taught me something important about financial prices: even without any fundamental news, prices can move because of the mechanics of liquidity provision and inventory management.

I also incorporated different types of traders into the simulator. This is important because real markets are heterogeneous. Not everyone follows the same strategy.

Some traders can behave more randomly, while others can respond to price movements or other market conditions. The interaction between these participants and the market maker can generate much more interesting price behaviour than a simple random walk.

This leads directly to one of the main ideas behind my project:

Chart patterns may not need to be magical shapes that somehow predict the future. They may simply be visual consequences of underlying market mechanisms.

If a certain combination of trader behaviour, order flow, liquidity, inventory pressure, or momentum repeatedly creates both a recognisable price structure and a particular future return distribution, then the chart pattern could contain genuine information.

That is a much more interesting explanation for technical analysis than simply saying that "a Head and Shoulders means price goes down."

Building different synthetic "worlds"
Another major development this week was expanding the simulator so that it can generate several different types of synthetic market environments.

I now think of these environments as different worlds.

Each world represents a different set of market assumptions or behaviours.

The reason this is important is that I do not want to construct one artificial market, discover that a particular chart pattern works inside it, and then conclude that the pattern is predictive.

That would tell me very little.

Instead, I want to ask a much more interesting question:

Under what market conditions do chart patterns become predictive?

This changes the entire purpose of the simulation.

Rather than attempting to reproduce the real stock market perfectly, I can create controlled experimental environments where particular mechanisms are stronger or weaker. I can then observe whether certain chart patterns emerge and whether their predictive ability changes.

This is similar to running experiments in a laboratory.

One of the improvements we implemented was allowing the program to combine the different worlds instead of forcing an entire simulation to exist inside only one environment.

The simulator can now combine as few as two worlds or as many as all nine worlds, depending on the experiment I want to perform.

This is particularly useful because real markets obviously do not remain in one simple regime forever.

A market might contain momentum behaviour during one period, stronger mean-reverting behaviour during another, different liquidity conditions later, and combinations of several effects at the same time.

Being able to mix the synthetic worlds therefore gives me a way of gradually increasing the complexity of my experiments.

I can start with very controlled situations and then progressively move toward much more complicated synthetic markets.

This also taught me an important research principle:

Complexity should be introduced gradually.

If I immediately create an extremely complicated simulator and discover an interesting result, it can become almost impossible to determine which mechanism produced the result.

By testing the worlds individually and then combining them, I can potentially identify which mechanisms are responsible for the predictive behaviour of particular patterns.

Improving the chart renderer
I also worked on the chart-rendering side of the project.

This initially seemed much less important than the simulator or pattern detector, but I realised that visualisation is actually extremely useful for debugging quantitative research.

The chart renderer allows me to visually inspect the synthetic price data that the simulator produces.

This matters because a program can run perfectly without throwing an error while still producing completely unrealistic data.

By looking at the generated charts I can ask questions such as:

Does this actually look like a market?

Is volatility sensible?

Are prices jumping in unrealistic ways?

Are the synthetic regimes visible?

Are patterns appearing naturally or because I accidentally forced them into the simulation?

The simulator and chart renderer therefore needed to be developed together.

I also learned how to run these scripts and open the resulting charts through the terminal rather than depending entirely on the VS Code interface.

This week made me increasingly comfortable working directly from PowerShell and using the terminal as a normal part of my development workflow.

Turning the Lo methodology into actual code
The other enormous part of this week was the pattern-detection algorithm.

Reading the Lo, Mamaysky and Wang paper was one thing. Translating its mathematical methodology into Python was much more difficult.

The central idea is that raw financial prices are noisy.

If I simply look for every tiny local maximum and minimum in raw prices, I will detect huge numbers of meaningless extrema caused by random fluctuations.

Therefore, the price series first needs to be smoothed.

I learned much more deeply this week how kernel smoothing works.

A nearby observation should generally have more influence on the estimated value than an observation far away from the point being estimated.

The Gaussian kernel gives us a mathematical way of assigning those weights.

The crucial parameter is the bandwidth.

A very small bandwidth follows the observed data extremely closely. This risks interpreting noise as meaningful structure.

A very large bandwidth creates an extremely smooth curve but can erase the actual structures that I am trying to detect.

This is therefore a classic bias-variance problem.

I also learned that the bandwidth should not simply be chosen because it "looks good."

The Lo paper used cross-validation to estimate an appropriate bandwidth and then multiplied the result by 0.3 because the cross-validated solution produced too much smoothing for their particular technical-pattern application.

At first I thought this meant that 0.3 was somehow the correct value.

I now understand that this is not true.

The optimal amount of smoothing can depend on the market, sampling frequency, volatility, window length, pattern being examined, and many other characteristics.

Therefore, blindly using a constant simply because it appeared in a research paper would be bad research.

The paper is a methodological starting point, not a set of universal constants.

This distinction has become very important to the direction of my algorithm.

Cross-validation and avoiding arbitrary parameters
I spent a significant amount of time understanding cross-validation more precisely.

The basic principle is surprisingly powerful.

Instead of estimating the smoothed curve using every observation and then judging how well it fits those exact same observations, I can temporarily remove one observation, estimate what its value should have been from the remaining observations, and calculate the error.

Doing this repeatedly gives an estimate of how well a particular bandwidth generalises rather than simply how well it fits the sample used to create it.

The bandwidth producing the lowest cross-validation error becomes a candidate for the optimal bandwidth.

This helped me understand a broader lesson which applies far beyond kernel regression:

Whenever I choose parameters using the same data on which I judge performance, I risk overfitting.

That is exactly the type of problem that can create apparently impressive trading strategies that disappear when they encounter new data.

Local-linear smoothing and boundary bias
During the implementation I also encountered an improvement over the simplest kernel-regression approach.

Instead of relying only on a local-constant smoother such as the basic Nadaraya-Watson estimator, the updated detector can use a local-linear smoother.

I learned why this matters particularly near the beginning and end of a data window.

In the middle of a dataset, a point usually has observations on both sides.

At the edge of a dataset, this is impossible.

A simple kernel-weighted average can therefore become biased near these boundaries.

Local-linear regression reduces this boundary problem because it locally estimates not only a level but also a slope.

This became particularly relevant because my detector uses rolling windows. Every time the algorithm reaches the most recent observation, it is effectively operating at the boundary of the available data.

That means boundary behaviour is not a minor mathematical detail. It can directly influence live pattern detection.

Understanding numerical precision and defensive programming
Another area in which I learned a surprising amount was numerical programming.

For example, I encountered a constant called _EPS, with a very small value such as (10^{-12}).

At first this seemed strange.

I learned that this does not mean the program thinks the real market contains quantities of (10^{-12}). It is simply a numerical safeguard.

Computers represent most decimal numbers using floating-point approximations.

As a result, calculations can occasionally create tiny rounding errors, divisions by numbers extremely close to zero, or values that are mathematically supposed to be equal but are not represented identically in memory.

An epsilon therefore prevents numerical instability.

I also learned why apparently small implementation details matter, such as rounding a bandwidth before using it as part of a cache key.

If two values differ only because of meaningless floating-point noise, treating them as completely different cached calculations wastes memory and computation.

This introduced me to the more general concept of deterministic computation.

For scientific research, it is extremely useful if the same inputs always produce exactly the same output.

Stable hashing, caching and reproducibility
I also examined code that generates stable hashes for arrays, labels, dates, and other pieces of data.

Originally, code dealing with byte representations, labels, isoformat(), struct.pack(), or floating-point arrays looked unnecessarily complicated.

I now understand why it exists.

If the program performs expensive calculations repeatedly on identical data, caching can dramatically improve performance.

However, the cache needs a reliable way of identifying when two inputs are genuinely the same.

This is why arrays and labels can be converted into stable byte representations and hashed.

I also learned that labels are not always simple integers.

A financial dataset could use integers, strings, Python datetime objects, Pandas timestamps, or other types as its index.

A robust program therefore should not assume that every label has the same representation.

This was a good example of the difference between writing code that works on my current test dataset and writing code that could eventually work on many different real-world datasets.

Data structures and Python concepts
A large part of this week was also spent improving my general Python knowledge.

I learned more clearly the difference between a list and a Sequence.

A list is one particular Python data structure.

A Sequence is a more general interface describing objects that behave like ordered sequences.

Using Sequence in function type hints therefore makes the program more flexible because the caller could potentially provide a list, tuple, NumPy-compatible structure, or another ordered sequence.

I also learned much more about:

dictionaries and mappings,
tuples,
sets,
dataclasses,
object attributes,
isinstance,
hasattr,
enumerate,
caching,
optional values,
type hints,
NumPy arrays,
axes in NumPy calculations,
and how functions communicate information through return values.
These concepts originally appeared to be small Python details, but I increasingly understand that good program architecture depends heavily on them.

For example, using a set of already-seen pattern identifiers can prevent duplicate pattern detections.

Using a dataclass such as a LocalExtremum object allows each detected turning point to carry structured information such as its type, position, and value instead of passing around unrelated variables.

Detecting extrema correctly
Pattern detection fundamentally depends on identifying local maxima and minima.

I originally imagined this as simply checking whether one price is higher or lower than its neighbours.

However, once a smoothed curve is involved, the mathematical interpretation becomes more interesting.

A local maximum occurs when the slope changes from positive to negative.

A local minimum occurs when the slope changes from negative to positive.

There is also the special case where the derivative becomes exactly zero for one or several observations.

In this situation, I learned that the algorithm cannot immediately decide whether the point is an extremum. It needs to look for the next non-zero derivative and determine whether the direction of the slope actually changed.

This made me appreciate why the apparently simple statement "find the peaks and troughs" becomes much more complicated when translated into robust code.

Mathematically defining chart patterns
Once the extrema have been found, the program still needs a mathematical definition of each chart pattern.

Humans can look at a chart and say "that looks approximately like a Double Top."

A computer cannot do that unless I explain precisely what "approximately" means.

For example, a Double Top requires two local maxima separated by a local minimum.

However, many further decisions need to be made.

How far apart can the two peaks be?

How similar do their heights need to be?

How deep should the trough between them be?

Does the second peak immediately count as a Double Top, or does price need to break the neckline before the pattern is confirmed?

How much tolerance is allowed?

These are not merely programming questions.

They are research definitions.

Changing them changes what the algorithm considers to be a pattern and therefore can change the research conclusion.

This taught me that algorithmic technical analysis forces vague chart-reading concepts to become explicit and testable.

That is one of the things I now find most interesting about the project.

Detection versus confirmation
One particularly important distinction I learned is the difference between the formation of a potential pattern and its confirmation.

Suppose the algorithm sees two peaks and a trough that geometrically resemble a Double Top.

At that point, a human looking retrospectively at a chart might already call it a Double Top.

However, many trading definitions require the price to subsequently break below the neckline before the pattern is confirmed.

The problem is that this information exists in the future relative to the second peak.

If I allow my algorithm to use future prices when deciding that the pattern existed earlier, I create look-ahead bias.

This is one of the most dangerous errors in financial backtesting.

I therefore worked on making the detector explicitly record detection and confirmation timing so that the program cannot accidentally pretend it knew something before the information was actually available.

This was one of the most important research lessons of the week:

Every prediction must be evaluated using only information that was available at the moment the prediction would actually have been made.

Rolling windows
I also improved my understanding of why the detector works using rolling windows.

Rather than passing an entire multi-year price series into the pattern detector and allowing the algorithm to analyse everything simultaneously, the detector examines a limited recent history.

For example, a window can contain approximately the previous 35 observations with an additional buffer where appropriate.

The window moves forward through time.

At every point, the algorithm behaves as if that point were the present.

This is important because it makes the experiment closer to a real trading environment.

The program should not be looking at tomorrow while pretending to make a decision today.

Pattern tolerances and market dependence
Another thing I realised this week is that there may not be one perfect set of pattern parameters for every market.

For example, deciding that two peaks must be within a particular percentage of one another might work reasonably well on one asset but poorly on another.

Cryptocurrency, equities, futures and foreign-exchange markets can have very different volatility structures.

Even the same market can move between low- and high-volatility regimes.

Therefore, I started thinking about tolerances in more statistically meaningful ways rather than treating every fixed percentage as universal.

This also connects to volatility.

I learned that volatility is closely connected to the standard deviation of returns or price changes, although the exact definition depends on what quantity and time scale are being measured.

That means a detector can potentially make some thresholds relative to the normal variability of the market rather than always using an arbitrary fixed number.

Robust aggregation
I also encountered the use of the median when combining several curves or estimates:

np.median(curves, axis=0)

This helped me understand why the median can sometimes be preferable to the mean.

If several estimates are being combined and one is extremely unusual, the arithmetic mean can be pulled strongly toward the outlier.

The median is much more resistant to this.

This introduced another recurring theme in quantitative research:

Robustness matters.

I do not want a single strange price observation, seed, simulation, or parameter to determine the conclusion of the entire experiment.

Expanding beyond an exact copy of Lo
Probably the most important design decision I made this week was that I do not want to create an exact reproduction of the Lo pattern-detection algorithm.

The Lo paper is incredibly useful because it gives me a rigorous framework for turning visual patterns into mathematical definitions.

However, my objective is different.

I want to build a strong modern detector that can operate both on my synthetic markets and eventually on real historical market data.

Therefore, I reviewed the algorithm repeatedly and looked for weaknesses that could affect my specific research question.

Instead of asking:

"Did I reproduce the paper exactly?"

I started asking:

"Is this the strongest and fairest methodology for my experiment?"

That is a much better research question.

Some of the improvements involved stronger input validation, clearer separation between detection and confirmation, more robust smoothing, safer handling of unusual data, reproducibility, better metadata, deduplication of detections, and designing the detector so that new patterns can eventually be added without rewriting the entire program.

Testing and debugging
Another lesson from this week is that debugging is not just fixing syntax errors.

A program can execute successfully and still be scientifically wrong.

Therefore, I started examining several different layers of correctness.

First there is normal software correctness:

Does the code run?

Are the data types correct?

Are array dimensions correct?

Do functions return what they are supposed to return?

Then there is numerical correctness:

Are calculations stable?

Can division by zero occur?

Can NaNs propagate through the program?

Are floating-point comparisons sensible?

Then there is financial correctness:

Does the market simulator behave plausibly?

Does the market maker respond correctly to inventory?

Do trader interactions make sense?

Then there is research correctness:

Is there look-ahead bias?

Are parameters being overfit?

Am I data snooping?

Am I evaluating patterns on the same data used to design them?

Would transaction costs destroy an apparent advantage?

This was probably the biggest development in how I think about programming.

I am no longer asking only:

"Does my code work?"

I am asking:

"Does my code correctly test the hypothesis I think it is testing?"

Those are very different questions.

Random seeds and Monte Carlo-style testing
I also returned to the idea of random seeds.

Using a fixed seed is extremely useful while debugging because it means that if something changes in the output, I know that the difference came from my code rather than simply from a different random simulation.

However, one seed obviously cannot be used to establish whether a result is statistically reliable.

Therefore, once the simulator and detector are stable, I want to run the experiment across many different seeds.

This connects directly to the Monte Carlo ideas I learned from the research paper.

If a pattern only performs well in seed 42 but fails across hundreds or thousands of other simulations, then the original result was probably luck.

If a relationship repeatedly appears across many independent simulated markets, it becomes much more interesting.

Synthetic markets as controlled experiments
I think I now understand much better why building the synthetic market before moving to real data is so useful.

With real historical data, I observe what happened, but I can never completely know the underlying process that generated every price movement.

With synthetic data, I control the rules.

Therefore, if I create a world containing a particular behaviour and suddenly a certain pattern becomes predictive, I have evidence connecting that underlying mechanism with the observed chart formation.

I can then remove the mechanism and see whether the predictive ability disappears.

I can strengthen it and see whether the effect grows.

I can combine it with other worlds and see whether the signal survives.

This potentially allows the project to investigate something deeper than:

"Do chart patterns work?"

The more interesting question could eventually become:

"Which market mechanisms cause particular chart patterns to contain predictive information, and under what conditions is that information strong enough to survive transaction costs?"

That feels like a much stronger research direction.

Real markets will still be the final test
Even if I discover strong relationships in my synthetic markets, I know that this will not automatically mean that the same relationships exist in reality.

The synthetic markets are experiments.

Real historical data will eventually be the test of external validity.

The long-term goal is therefore to run the same detector on real market data without changing the rules after seeing the answer.

Ideally, parameters should be determined using training or validation data and then evaluated on genuinely unseen out-of-sample data.

I also want to test different types of markets rather than assuming that results from one asset apply everywhere.

The research I read earlier suggested that technical-analysis opportunities may behave differently depending on market efficiency, liquidity, competition, and transaction costs.

My simulator may eventually give me a way of investigating why.

Git, GitHub and version control
Alongside all of the quantitative work, I have also learned a surprising amount about Git and GitHub.

At the beginning of the month, GitHub was almost completely unfamiliar to me.

I initially thought of it mainly as somewhere to upload my code.

I now understand that Git is really a version-control system.

Every commit creates a record of the state of my project.

That allows me to experiment while preserving the history of what I changed.

I have learned commands and concepts such as:

git status

git add

git commit

git pull

git push

remote repositories,

origin,

branches,

main,

rebasing,

and conflicts between local and remote histories.

I encountered several real problems while doing this.

For example, I had situations where my local repository and the GitHub repository had different histories.

Git refused to push because GitHub contained commits that my computer did not yet contain.

I learned why Git does this: it is protecting the remote repository from having work accidentally overwritten.

Instead of simply forcing my version onto GitHub, the correct solution is normally to fetch or pull the remote changes, integrate them, resolve any conflicts, and then push the combined history.

I also encountered a network error where Git reported:

Could not resolve host: github.com

This taught me that not every Git error is actually caused by Git. That particular problem was a DNS/network-resolution problem.

Once the computer could contact GitHub again, Git returned a completely different error concerning the repository history.

Learning to distinguish networking problems from Git problems is another small but useful skill I gained.

I also now understand why forcing a push can be dangerous.

A command can technically make the error disappear while simultaneously deleting valuable remote history.

Version control is therefore another area where understanding why something happens is much more important than memorising commands.

Organising the research repository
I also organised the research papers I have been reading inside the project so that the code and the academic foundations of the research remain connected.

This is useful because I want each major methodological decision to have a reason behind it.

Rather than randomly adding techniques because they sound sophisticated, I can trace ideas back to papers, compare different methodologies, and explain why I accepted or rejected them.

I think this will become particularly important when I eventually write the final research report.

What I learned most this week
Looking back, the biggest thing I learned this week is that quantitative research lies at the intersection of several completely different disciplines.

I need programming because the experiments need to be implemented correctly.

I need mathematics and statistics because I need to know whether the results mean anything.

I need finance because the simulated market needs to have economically sensible mechanisms.

I need research methodology because it is extremely easy to accidentally create biased results.

And I need software-engineering practices such as testing, reproducibility, version control and documentation because the project is already becoming too large to manage casually.

Something else I learned is that understanding the code is much more valuable than simply having working code.

Throughout this week I repeatedly stopped on individual lines and asked why they existed.

Sometimes it was something tiny, such as why a function used a tuple instead of a list, why an epsilon was (10^{-12}), why an array was converted into bytes, why axis=0 appeared in a NumPy function, or why an object needed a particular attribute.

Those questions sometimes slowed the development down considerably, but they also changed the project from code that I possessed into code that I actually understood.

I think this is extremely important because eventually I need to defend the methodology.

If somebody asks why I used a particular smoother, bandwidth, tolerance, window length or statistical test, "because the program generated it for me" is not an acceptable answer.

I need to understand every important assumption.

Where the project stands now
At the end of this week I now have the foundations of the three major technical components of the project:

A substantially more sophisticated synthetic market simulator, including a market maker, multiple trader behaviours, different synthetic market worlds, and the ability to combine several worlds together.

A chart-rendering system that allows me to visually inspect and debug the artificial markets being generated.

A much more advanced pattern-detection framework, inspired by the Lo methodology but being redesigned for my own experiment, with kernel smoothing, bandwidth selection, extrema detection, mathematical pattern definitions, rolling-window analysis, confirmation timing and protections against look-ahead bias.

There is still a lot to do.

The detector needs to be tested extensively.

More pattern definitions need to be implemented and validated.

The synthetic experiments need to be run across many seeds and parameter configurations.

The resulting returns need to be analysed statistically.

Transaction costs need to be incorporated before calling anything economically profitable.

Eventually the system needs to be tested on real historical data using proper out-of-sample methodology.

However, the project now feels fundamentally different from where it was one week ago.

I am no longer simply trying to program a chart-pattern finder.

I am beginning to build an experimental framework for studying why, when, and under what market conditions technical chart patterns could contain information about future prices.

That distinction may ultimately become the most important idea behind the entire project.


19/08/2026 --

Today I concentrated on turning the detector into something that could actually support a proper backtest rather than just print pattern names.

The biggest change was thinking much more carefully about the difference between a raw pattern candidate and a genuine causal event.

Because I use several smoothing scales and rolling windows, the same underlying chart formation can potentially be rediscovered several times. If I counted every one of those detections as an independent trade, I would artificially inflate the sample size and make the statistical evidence look much stronger than it really is.

So I worked on clustering and deduplicating detections. The detector now keeps the raw multi-scale candidates for research diagnostics, but the important output for trading research is the set of unique causal events.

This gave me a much cleaner research contract:

- candidates = all raw pattern views found across scales
- clusters = groups of detections that appear to represent the same underlying event
- events = the unique causal pattern occurrences that can actually be passed to a backtester or AI model

I also made the timing semantics explicit.

Each pattern records when the pattern geometry ended, when enough information existed for the detector to make the event available, and the earliest later candle at which a conventional backtest is allowed to execute.

This made the detector much more useful because the downstream code no longer has to guess when a pattern should become tradeable.

I also reinforced the rule that the detector itself never decides whether a pattern was profitable. It only describes the geometry and its classical expected direction. Future prices remain completely outside the detector.

This separation is becoming one of the central design principles of the project.


21/08/2026 --

Today I worked on the market-structure experiment and on the first serious real-market backtests.

The synthetic side now has nine different market worlds, which means there are 511 non-empty combinations of worlds that can be tested.

The idea is not simply to generate one complicated artificial market and search for a lucky result. I want to systematically compare many possible market structures.

I therefore built the experiment so that it can:

- enumerate the non-empty combinations of worlds
- assign market weights
- run independent seeds
- build candles using the same candle constructor
- run the exact same frozen pattern detector
- enter only after the event is actionable
- test predefined future horizons
- include transaction costs
- compare pattern outcomes with null opportunities
- aggregate results across independent runs rather than pretending every pattern inside one path is independent

I also learned more about multiple-testing problems.

If I test hundreds of structures, many patterns and several horizons, eventually something impressive will happen by chance.

Therefore the experiment includes statistical corrections rather than simply reporting the biggest raw return.

I incorporated discovery and validation logic so that the best-looking structures in the exploratory stage still have to survive on fresh seeds.

The backtesting cost model also became more realistic.

The result is not just:

future price - entry price

It accounts for bid/ask spread, brokerage and optional slippage.

This is important because a pattern can be statistically directional while still being economically useless after trading costs.

I also started testing the exact same detector on real BTC and ETH historical data. This is important because synthetic results are only controlled experiments. Real markets are the external test.


23/08/2026 --

Today was mainly about preparing the project for the Polymer Technology Exposition submission.

The deadline forced me to compress a very large project into a very short explanation.

I created the one-page project write-up and prepared the project-showcase section of the video.

The research question I settled on was:

"Under what market conditions does pattern trading have a predictable edge?"

I think this is much stronger than simply asking whether technical analysis works.

The one-page summary explains the project as a sequence:

1.) detect chart patterns mathematically
2.) create controlled synthetic markets
3.) vary the market structure
4.) backtest patterns after realistic costs
5.) use AI to learn the context surrounding successful patterns
6.) compare the synthetic findings with real BTC and ETH markets

I also used the application as an opportunity to explain how AI was used in the project.

The AI is not being used to magically draw a Head and Shoulders pattern on a chart.

The classical detector identifies the chart formation first.

The AI then asks:

"What else was true about the market when this particular occurrence succeeded?"

I also had a small preliminary AI experiment available for the application.

The preliminary dataset contained 2,628 detected patterns across about 300 synthetic structures.

Before AI filtering, approximately 44.56% of those patterns were profitable and the average result was about -0.149%.

In a very small unseen sample, an early MLP filter selected trades with about a 70.7% win rate and around +0.816% average net return per selected trade.

The unseen sample was only 41 trades, so I made it very clear that this was a small test and not a final research result.

The main value of that early experiment was that it supported the idea that context may matter much more than the pattern name by itself.


25/08/2026 --

I spent more time analysing the real BTC and ETH backtests.

This stage has been useful because it immediately showed that a result can change enormously depending on the candle timeframe.

The very short timeframes were generally disappointing after transaction costs.

The overall picture I had was roughly:

- 1-minute results were negative
- 15-minute results were negative
- 30-minute results were mostly negative
- 1-hour results were more promising
- daily results were much stronger in some cases but had much smaller samples
- weekly results were too sparse to make a strong conclusion

For the 20-bar all-pattern tests, the 1-hour results included approximately:

BTC: +0.163% average net return per accepted trade
ETH: +0.118% average net return per accepted trade

The daily results were larger but much less statistically comfortable because there were fewer trades.

One result that stood out was the BTC daily Double Bottom at the 20-bar horizon, which produced around +7.686% average return per trade across 16 trades.

Sequentially compounding those individual accepted trades gave a very large historical number, but I learned that this has to be interpreted extremely carefully.

A sparse strategy that is only invested during selected periods cannot be compared directly with a continuously invested benchmark without discussing the exposure difference.

I therefore compared the rough magnitude with the S&P 500 price index over the same broad August 2017 to August 2026 period, while noting that S&P dividends were excluded and that continuous benchmark exposure is fundamentally different from intermittent pattern exposure.

This exercise reinforced something I already suspected:

A large historical return is not automatically convincing evidence.

Sample size, exposure, costs, selection bias and out-of-sample behaviour matter just as much as the headline number.


28/08/2026 --

Today I spent time analysing the synthetic market-structure results more deeply.

The most interesting structure from the earlier screening was approximately:

60% rule-based traders
20% emotional traders
20% mean-reversion traders

At the 20-bar horizon, this structure had produced roughly +1.215% average net return per accepted trade in the earlier experiment.

The result itself was interesting, but what I found more valuable was trying to understand why this particular combination could produce pattern-like behaviour.

My current interpretation is:

Rule-based traders can extend trends or react to mechanical signals.

Emotional traders can exaggerate moves through FOMO, fear and other feedback.

Mean-reversion traders can then push prices back toward a reference or perceived fair value.

The interaction can therefore create a push-pull sequence.

A move can extend, overshoot, correct, retest and then reverse again.

Those are exactly the types of dynamics that can create shoulders, repeated highs or lows, triangles and other classical chart formations.

This gave me a much more convincing explanation for why patterns could sometimes contain information.

The shape itself may not be the cause.

The shape may be a visible summary of a deeper market mechanism.

I also reinforced that this "winning" structure cannot be trusted because of one simulation.

It needs to be tested on fresh seeds and eventually compared with unseen structures and real markets.


30/08/2026 --

Today I designed the AI research pipeline much more carefully.

The AI dataset is built downstream of the frozen classical stack:

market simulator
-> candle builder
-> pattern detector
-> AI research pipeline

I deliberately did not create a second AI-specific detector.

Every AI observation comes from the same causal pattern events produced by the classical detector.

The target of the model is approximately:

P(pattern is profitable after costs | pattern geometry, observable market state, market structure)

I divided the inputs into two major feature sets.

Observable features are things that could potentially exist in a real trading environment.

Examples include:

- pattern type
- expected direction
- geometry fit score
- pattern length
- pattern height
- breakout strength
- recent returns
- volatility
- trading range
- volume
- buy/sell imbalance
- order-flow persistence
- spread
- liquidity proxies
- price-impact proxies

Then I created an Oracle feature set.

Oracle contains the Observable features plus information that only exists because the market is synthetic.

Examples include:

- the true weights of the nine market worlds
- whether the structure is a random null
- the simulator's true spread and volatility settings
- emotional FOMO and fear sensitivity
- informed-trader share
- mean-reversion strength
- momentum strength
- regime persistence
- liquidity state
- adaptive learning settings
- hidden sentiment and information variables

The purpose of Oracle is not to create a deployable trading model.

It is a mechanism-discovery tool.

If Oracle learns a strong relationship between a hidden synthetic mechanism and pattern success, that gives me a hypothesis that I can later try to approximate using observable real-market variables.


01/09/2026 --

Today I worked on one of the most important AI safeguards: how the dataset is split.

It would be very easy to randomly split individual pattern rows into training and testing.

However, this could create a serious form of leakage.

Patterns generated by the same synthetic market structure may share many hidden characteristics.

If one pattern from Structure A is in the training set and another pattern from Structure A is in the test set, the model may partly recognise the structure instead of truly generalising to a new environment.

Therefore I split entire market structures.

The planned split is:

60% training
20% validation
20% untouched testing

All seeds belonging to one structure stay in the same partition.

This means that:

Structure A, seed 1
Structure A, seed 2
Structure A, seed 3

must all belong to the same split.

This was an important conceptual improvement.

A seed is not a new market structure.

It is simply another random realisation of the same underlying mechanism.

The validation partition is used for model and probability-threshold decisions.

The test partition is supposed to remain untouched until the analysis choices have already been made.

This is much closer to how I would want to evaluate a real quantitative model.


03/09/2026 --

Today I focused on the actual AI models and research outputs.

I kept three model families because they each answer a slightly different question.

Logistic Regression is the simple baseline.

If a complicated model cannot beat a basic linear model, that is useful information.

XGBoost allows nonlinear interactions and is especially useful because I can inspect feature importance and SHAP values.

The MLP gives me a neural-network comparison.

I also built the pipeline so that model quality is not judged only using classification accuracy.

In trading, a model can be statistically accurate but economically useless.

The pipeline therefore also measures things such as:

- precision
- recall
- ROC-AUC
- average precision
- calibration
- Brier score
- selected-trade count
- profitability after costs
- average net return
- results by pattern type

The probability threshold is selected on validation data.

For example, instead of automatically trading every model prediction above 50%, the validation set can determine whether a stricter threshold produces a better economic trade-off.

I also added explicit protections around future information.

Columns containing realised profitability, future directional return, entry and exit outcomes, favourable excursion and adverse excursion are treated as outcome fields and are not allowed to become AI input features.

Future prices are allowed to answer:

"Did the pattern succeed?"

They are not allowed to answer:

"What should the AI have known before the trade?"


05/09/2026 --

I spent time improving reproducibility and scaling.

The project is now large enough that simply remembering which version of the detector produced which dataset is not acceptable.

I therefore worked on provenance.

The research pipeline stores information such as:

- simulator version
- random-stream version
- candle-builder version
- detector version
- experiment-runner version
- configuration hashes
- source-file hashes
- dataset fingerprints
- structure IDs
- scenario IDs
- run IDs
- random seeds

The AI pipeline also freezes an experiment manifest.

This means that if I generate a dataset and later change the detector or research code, the training command can detect that the current code no longer matches the frozen experiment.

Instead of silently training incompatible data, the program is designed to fail loudly.

I also improved the computational side of the project.

Dataset generation can use multiple processes because thousands of structures and seeds would otherwise take an unreasonable amount of time.

At the same time, I learned that blindly combining multiprocessing with libraries that create their own CPU threads can actually make performance worse.

This is why the project uses thread limits around some numerical libraries.

The goal is not simply "use as many CPUs as possible."

The goal is controlled parallelism without every process starting its own large group of threads.


07/09/2026 --

Today I carried out another very detailed audit of the complete pipeline.

This was partly because I realised that the most dangerous research bugs can exist between files rather than inside one file.

For example, the simulator could produce perfectly valid data and the detector could work perfectly on its own, but the final results would still be wrong if the backtester misunderstood the detector's timing.

I therefore checked the interaction between:

market_maker_simulator.py
chart_renderer.py
pattern_detector.py
market_structure_experiment.py
ai_pattern_research_pipeline.py

I focused on questions such as:

Does every part of the project use the same candle construction?

Does the detector ever use future prices?

Does the backtester enter only after earliest_execution_index?

Are raw multi-scale candidates accidentally being counted as independent trading signals?

Are transaction-cost calculations consistent?

Are the AI labels calculated using the same economic-return assumptions as the classical backtest?

Can structure IDs leak between training and test?

Can future/outcome columns accidentally enter the model feature list?

Are seeds and source-code versions recoverable later?

This audit was extremely useful.

It also forced me to understand many smaller pieces of the simulator much more deeply.

I spent time learning things such as:

- what mark price represents
- how the public fundamental differs from the latent fundamental
- how synthetic traders generate signals
- what z-scores mean
- why some scales use square roots
- how round-number liquidity effects work
- what return scales represent
- how geometric Brownian motion updates work
- how inventory changes market-maker quotes

This may have slowed down development, but it made me much more confident that I can explain the code rather than simply show it.


08/09/2026 --

Today I focused on how the AI experiment is actually run from the terminal.

I clarified the difference between structures and seeds because I initially confused the two.

If the experiment has:

1000 structures
and 3 seeds per structure

that means 3000 synthetic runs.

It does not mean 3000 independent market structures.

Each structure has a frozen set of market-generating conditions, while the seeds create different random paths under those same conditions.

This is useful because repeated seeds help test whether a relationship is stable instead of being caused by one lucky path.

I also clarified the AI pipeline commands.

The pipeline can:

build-dataset

which generates the synthetic structures, simulates the runs, builds candles, detects patterns and creates the labelled AI dataset.

train

which loads an already frozen dataset and trains/evaluates the models.

all

which performs both stages.

This is much better than repeatedly clicking "Run" without understanding what stage is being executed.

I also revisited Monte Carlo terminology.

The Monte Carlo mode inside market_maker_simulator.py only repeats the simulator across seeds and averages market statistics.

It does not run the pattern detector.

Pattern-level repeated-seed research happens in market_structure_experiment.py and ai_pattern_research_pipeline.py.

That distinction is important because there are now several different ways of repeating simulations in the project and they answer different questions.


09/09/2026 --

Today I prepared the project for the Polymer Technology Exposition tomorrow.

The main goal was to make sure I can explain the project quickly but also go into technical detail if somebody asks.

I organised the material into a presentation/demo structure:

- submitted one-page Polymer project overview
- source code
- synthetic charts
- pattern detections
- real-market backtest results
- AI results and datasets
- research papers
- project journal
- terminal command reference
- presenter guide

I realised that my submitted one-page PDF already works well as the high-level architecture overview.

It has the full research flow:

Detect -> Simulate -> Discover -> Backtest -> AI

So rather than showing another almost identical architecture diagram immediately afterwards, I plan to use the submitted PDF as the overview and move directly into the working project.

I also designed a very short live demo.

The current plan is:

1.) generate a synthetic market using the winning 60/20/20 structure
2.) open the generated OHLCV chart
3.) pass the generated candle data through the pattern detector
4.) show a detected Head and Shoulders example
5.) point out when the pattern became available
6.) point out the first allowable execution candle
7.) explain that only after this point do future prices create the success/failure label
8.) connect that occurrence to the AI dataset
9.) show that the AI learns the conditions surrounding successful and unsuccessful occurrences

I also created a small run_pattern_detector.py wrapper for the demo because pattern_detector.py itself is primarily a Python module and does not currently expose a convenient CSV command-line interface.

This means the demo can be much cleaner:

python market_maker_simulator.py ...

followed by:

python run_pattern_detector.py results\synthetic_market_candles.csv

I searched deterministic seeds in the winning synthetic structure to find a clean H&S example that will be easy to explain visually.

Seed 41 was particularly useful.

The detector found a clear Head and Shoulders with approximately:

left shoulder: 102.83
first neckline trough: 99.76
head: 103.48
second neckline trough: 99.89
right shoulder: 102.90

The shoulders are almost perfectly aligned and the neckline troughs are also close to one another, making it a very clear presentation example.

The pattern occurs roughly between candles 416 and 462.

The detector only makes the event available later, and the earliest permitted execution is later again.

The price then moves lower afterwards, which makes it a useful example for explaining the complete causal workflow.

I also created a terminal-command guide containing the major arguments available across:

market_maker_simulator.py
pattern_detector.py
ai_pattern_research_pipeline.py

This helped me consolidate what can actually be controlled through the terminal and what remains a programmatic research parameter.

Looking back at everything written in this journal, the project has changed enormously.

It started with a simple random market and a chart.

It has now become:

random/null markets
-> market maker and order flow
-> nine synthetic worlds
-> configurable world mixtures
-> OHLCV construction
-> kernel-smoothed causal pattern detection
-> unique event clustering
-> execution-safe backtesting
-> transaction costs
-> matched nulls and statistical testing
-> 511 market-structure combinations
-> repeated seeds
-> BTC/ETH real-market backtests
-> AI pattern-context dataset
-> Observable and Oracle feature sets
-> Logistic Regression / XGBoost / MLP
-> structure-level train/validation/test splits
-> explainability and SHAP
-> reproducible manifests and source fingerprints

The most important thing I have learned is that finding a chart pattern is the easy part.

The difficult part is proving that the apparent edge is not caused by:

- chance
- transaction costs being ignored
- look-ahead bias
- data snooping
- overlapping or duplicated detections
- multiple testing
- lucky random seeds
- training/test leakage
- choosing parameters after seeing the answer

That is probably the biggest change in how I think about quantitative finance.

At the beginning I wanted to know whether chart patterns "work."

Now I think the better question is:

"Under what market conditions does a specific detected chart pattern contain enough information to produce a repeatable edge after costs?"

That is the question I want the final project to answer.
