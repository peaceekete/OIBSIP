# Customer Segmentation Analysis

## What this project does

This project groups an online retailer's customers into five behavioral segments based on how recently they bought something, how often they buy, and how much they spend. The goal isn't the clusters themselves, it's turning those groups into specific actions a business could actually take: who to win back, who to reward, who to nudge toward a second purchase, and who to stop spending marketing budget on.

## Dataset

I used the Online Retail II dataset from the UCI Machine Learning Repository, covering a UK-based online gift retailer's transactions from December 2009 to December 2011. It contains just over a million transaction rows, with a customer ID, invoice date, quantity, price, and product description for each line item.

I picked this for a few reasons. It covers two full years instead of one, which matters a lot for Recency and Frequency, those numbers only mean something if there's enough time in the data for patterns to show up. And it comes from the same research group that published the original paper on RFM-based segmentation for online retail, so there's a solid academic basis for using it this way.

## Cleaning the data

The raw data had a fair number of problems that needed sorting out before any of the RFM numbers could be reliable.

**Cancellations and invalid rows.** About 19,500 rows were cancelled orders, marked with a "C" prefix on the invoice number, and around 23,000 had negative quantities. I excluded anything that was a cancellation, had a non-positive price, or had a negative quantity, along with roughly 243,000 rows that had no customer ID at all (you can't segment a customer you can't identify).

**Duplicate rows.** After that first cleanup, about 26,000 rows turned out to be exact duplicates, same invoice, same item, same quantity, same price, same timestamp. These looked like export artifacts rather than real repeat purchases, and they were adding up to about £368,600 in inflated spend across the dataset. I dropped them.

**Non-product line items.** A few stock codes in the data weren't actual products at all. Things like "POST" (postage), "M" (manual charges), "DOT" (Dotcom postage), "ADJUST" (account adjustments), and bank charges. One of these turned out to be a real problem: a customer had a manual charge of nearly £9,000 on one invoice, which was then fully reversed on the very next invoice as a cancellation. Because my filter dropped cancelled invoices but kept the original manual charge, that customer would have shown up with artificial £9,000 contribution to their monetary value that never actually happened. I removed all of these non-product codes to avoid that kind of distortion.

After all of this, the dataset went from 5,942 raw customer IDs down to 5,852 customers with a genuine, cleaned purchase history to build RFM from.

## Building the RFM features

For each of the 5,852 customers, I calculated:

* **Recency**: days since their most recent purchase, measured from the day after the dataset's last recorded transaction
* **Frequency**: number of distinct invoices (unique purchase occasions)
* **Monetary**: total amount spent, calculated as quantity multiplied by price, summed across every valid transaction

## Getting the data ready for clustering

Before running K-Means, I checked whether these three features were skewed, since K-Means relies on distance calculations that get thrown off by long tails of extreme values.

Recency came out only mildly skewed (0.89) and didn't need any transformation. Frequency and Monetary, on the other hand, were extremely skewed (12.03 and 25.33), which meant a small number of very high-frequency, very high-spending customers were stretching the whole distribution. I applied a log transform to both, which brought Monetary down to 0.27 and Frequency down to 1.00, a big improvement, even if Frequency still carries some skew naturally since it's count data with a hard floor at 1.

After that, I scaled all three features with StandardScaler, so that Recency's much larger raw numbers (ranging into the hundreds) didn't dominate the distance calculations just because of their size, rather than their actual importance.

## Choosing the number of clusters

I ran K-Means across K values from 2 to 10 and checked two things: the elbow method (how much adding another cluster reduces inertia) and the silhouette score (how well separated and tight the clusters are).

The elbow curve didn't have one single obvious bend, the slowdown in inertia reduction was gradual between K=4 and K=5 rather than a sharp corner. Because of that, I used the silhouette score as a tiebreaker, and it showed something worth noting: scores generally decline as K increases, but K=5 scored slightly higher than K=4 (0.365 versus 0.361), the only point in the whole curve where the trend reversed. K=5 was selected as a practical balance between cluster separation and business interpretability.

## What the five clusters look like

|Cluster|Recency (days)|Frequency|Monetary (£)|Customers|
|-|-|-|-|-|
|0|394.29|3.35|1,262.16|845|
|1|63.52|5.75|1,937.22|1,688|
|2|41.46|22.98|13,379.13|890|
|3|97.53|1.73|412.84|1,350|
|4|518.88|1.19|248.95|1,079|

**Cluster 0, lapsed mid-value customers.** These customers built a real purchase history, several orders and a decent amount of spend, but haven't bought anything in over a year.

**Cluster 1, active and reliable.** Recent, moderately frequent buyers with solid spend. These are the customers currently engaged with the business.

**Cluster 2, high value, but likely a mix of individuals and wholesale buyers.** This group spends far more than anyone else, roughly seven times the next highest cluster, and their average order frequency (nearly 23 orders, with some customers over 300) is well beyond what a typical individual shopper would do. Plotting frequency against spend on a log scale showed this cluster's numbers climbing together in a tight line, a pattern that looks more like wholesale or resale buying than individual loyalty. This is worth treating carefully rather than assuming it's simply "our best customers."

**Cluster 3, new customers.** Their median first purchase date was mid-2011, noticeably later than every other cluster, and I checked this directly rather than assuming it from the RFM numbers alone. They've made a purchase or two but haven't had the chance yet to become repeat buyers.

**Cluster 4, dormant.** These customers have been inactive for about 70% of the entire two-year window covered by the dataset, with very low frequency and spend even when they were active. The dataset doesn't have an actual churn label, so this is an inference based on how long they've been inactive relative to the observation window, not a confirmed fact, but it's a strong enough pattern to act on.

## What I'd actually recommend doing with each group

**Cluster 0 (lapsed, mid-value):** a win-back campaign, ideally referencing what they bought before, since they've already shown they're willing to spend, they just went quiet.

**Cluster 1 (active, solid value):** enroll them in a loyalty or retention program before they have a chance to drift into Cluster 0's pattern.

**Cluster 2 (high value, mixed):** don't treat this as one VIP segment. Split it further, flag the highest-frequency accounts (say, above 50 orders) for review as potential wholesale or business accounts and route them toward account management or bulk pricing, while treating the remaining high-spending individuals with genuine VIP perks.

**Cluster 3 (new):** second-purchase incentives and personalized recommendations based on their first order. The goal here is converting them into repeat buyers, not retaining a relationship that hasn't formed yet.

**Cluster 4 (dormant):** don't spend much here. If anything, a single low-cost automated win-back email, not a full campaign, since the odds of reactivation are low and their historic value doesn't justify more.

## Limitations

This dataset is over a decade old, and while the RFM method itself still applies to any retailer, the specific numbers here are historical and wouldn't transfer directly to a current customer base without redoing the analysis on recent data.

There's no field in the data that separates individual consumers from business or wholesale buyers, which is exactly why Cluster 2 is ambiguous. A production version of this analysis would need that information to segment reliably.

There's also no confirmed churn label anywhere in the dataset. Every mention of "churned" or "at risk" in this project is an inference based on how long a customer has been inactive, not a measured outcome. A customer with high Recency might come back after the dataset ends, there's simply no way to know from this data alone.

K-Means itself comes with assumptions worth keeping in mind: it expects clusters to be roughly round and similarly sized, and it's sensitive to both the initial starting points and the choice of K. The elbow curve here didn't have one clean, obvious answer, K=5 was the best-supported choice given the evidence, not a mathematical certainty.

And RFM only measures how much and how often someone buys, not what they actually buy. Two customers with identical RFM numbers could have completely different tastes, so these segments support broad engagement decisions (win-back, loyalty, retention) but not product-level recommendations.

## Tools used

Python, pandas, numpy, matplotlib, and scikit-learn (KMeans, StandardScaler, silhouette\_score), all run in Jupyter Notebook.

## How to reproduce this

1. Download the Online Retail II dataset from the UCI Machine Learning Repository.
2. Run the notebook top to bottom. It loads both sheets of the dataset, cleans out cancellations, invalid rows, duplicates, and non-product line items, builds the RFM table, applies the log transform and scaling, runs K-Means with K=5, and produces the cluster profiles and visualizations shown above.

