## Use cases
- Identify ideas that are implemented and used (validated), but users are not satisfied with them (bad average rating).
- Deduce problems from negative app reviews.
- Verify if a problem is present for other people.

## Tech stack
- chromadb (all-MiniLM-L6-v2)
- python
- sklearn (DBSCAN, KMeans)
- selenium

## Conclusions
- So far, at about 80000 scraped relevant negative reviews with ratings 1 and 2, the clusters suggest problems such as: 
    - Subscriptions cannot be canceled;
    - Too many ads; 
    - Paid app features do not activate;
    - Specific app issues;
- At ~4000 analyzed apps and clustered by the embeddings of their descriptions, the results suggest that the apps with the lowest average rating are in the following categories: file sharing/transfer, virtual phone numbers, news, audiobooks.