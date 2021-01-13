# Voice of the Customer

## Overview

As an Amazon Seller, or really anyone who sells product, it's important to always be on the lookout for opportunities to fill a niche or an unment consumer demand. In that vein, a seller's number one priority should aways be the customer. In this project, I set out to develop a repeatable process for ingesting, analyzing, and applying consumer feedback to the product discovery process.

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/movers.png'>

To accomplish this, I built a web scraper for collecting reviews associated to any customer search term on Amazon that could then be run through a data cleaning and natural language processing pipeline to be fit to a Latent Dirichlet Allocation model for discovering topics in both positive and negative reviews. The final product is an application built with Plotly Dash that accepts a csv file of Amazon Customer Reviews to automate the cleaning, NLP, and visualization of sentiment and topics.

### Technologies

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/flow.png'>

## Data

The core data used for this analysis are reviews scraped from Amazon's US marketplace using a Python script leveraging the requests and BeautifulSoup libraries, hosted on an EC2 instance for convenient IP rotation when the inevitable blocker appears (i.e. captchas - the workaround for these is beyond the scope of this project). The focus of the presentation was on the top products for the term "portable air conditioner".

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/review.png'>

The "portable air conditioner" term was selected due to the presence of an ASIN in the Amazon Movers and Shakers list. A preliminary look at results for the term revealed a significant number of poorly reviewed products owning top presence in search. This is generally a good indicator of market opportunity on Amazon. A deeper look at the customer reviews scraped from the top products associated to this term showed a significant number of 1-star reviews - though a look at the histogram below showing the distribution of reviews does not immediately raise a flag, as it appears there are far more positive reviews. The sentiment analysis using Textblob reveals a much larger disparity between positive and negative.

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/sentiment.png'>

Even more important is a deeper dive into the rating of products over time, as incentivized 5-star reviews are generally a part of the initial launch of the product. This practice artificially inflates perceived consumer sentiment. The below moving average review rating reveals customer dissatisfaction increases over time.

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/average.png'>

## Model and Results

The corpus was created through a concatenation of the review titles and body text to ensure all verbage was captured. Preprocessing was done to remove punctuation and special characters prior to lemmatization and the inclusion of n-grams to more accurately capture phrasing. Documents are then converted to TF-IDF vectors for model fitting. Latent Dirichlet Allocation is the model of choice for this project, as the use of probabilities of terms appearing together within documents and topics is fitting for the use case. The chart below shows how Sklearn's GridSearchCV is used with LDA to deterine the best fit of a 0.5 learning decay and 3 topics - we are looking for the highest log-likelihood.

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/train.png'>

The PyLDAvis library is used to provide a more intuitive visual of the results. The process was repeated for both positive and negative reviews separately to get a more solid understanding of what product features led to positive reviews for inclusion in future product iteration and what features led to customer dissatisfaction.

<b>Positive Review LDAvis:</b>

This helps to provide further context to what common themes occur within positive reviews. Another step closer to understanding the customer, as you can see that positive reviews highlight the product cools as expected and mentions the use of the water cooling system used in many of these products. The same general insight can be gleaned from the negative reviews as well.

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/pos.png'>

<b>Negative Review LDAvis:</b>

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/neg.png'>

However, what makes this even more useful in practical application is to look at the most heavily weighted reviews within each topic. These have the highest probability of appearing together and within the topic based on the content found within. This becomes a truly time saving activity because we now do not have to sift through hundreds or thousands of reviews, and we still get the richness of reading the reviews directly from the customer. Take that, useless word clouds.

After completing this analysis, the required visualizations and data transformations were incorporated into an app powered by Plotly Dash. In its current state, it runs locally for use within personal projects, though a public-facing version will be launched in the near future.

<img src='https://github.com/ryankirkland/voice-of-the-customer/blob/master/img/app.png'>
