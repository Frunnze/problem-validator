## Use cases
- Find app ideas that are implemented and used (validated), but people are not satisfied with them (bad average rating).
- Deduce problems from negative app reviews (of one or more apps).
- Verify if a problem is present for other people.

## Tech stack
- chromadb
- python
- sklearn
- selenium
- all-MiniLM-L6-v2

## Conclusions
- So far, at about 80000 scraped relevant negative reviews with ratings 1 and 2, the clusters suggest problems such as: 
    - Subscriptions cannot be canceled;
    - Too many ads; 
    - Paid app features do not activate;
    - Specific app issues;