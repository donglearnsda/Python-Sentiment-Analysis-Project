# Water-Sort-Puzzle-Project
The project using Python to API review data from CH Play and AppStore of the game "Water Sort Puzzle" and doing EDA, sentiment analysis with this dataset.
1. What is sentiment analysis?

Sentiment analysis is the process of analyzing text to determine the sentiment expressed, classifying it as positive, negative, or neutral. It helps understand the subjective opinions, emotions, or attitudes conveyed in textual data.

2. Project goals:

Having better understanding of customers' opinions, making data-driven decisions based on review sentiment.

3. Setup process:
- Step 1: Collect review data from CH Play and AppStore
- Step 2: Concatenating results between CH Play and AppStore
- Step 3: Doing sentiment analysis
  - Clean and preprocess the text data by converting the text to lowercase and removing stopwords
  - Topic modeling using **LDA** by classifying dataset into 3 topics
  - Make **Wordcloud** of positive/negative review
  - **VADER sentiment scoring** include running the polarity score on the entire dataset and plotting VADER result

4.  References:
- "Sort Water Puzzle" publishing company: https://www.sonat.vn/
- NLP on ActiveSG Reviews (Kaggle): https://www.kaggle.com/code/jiayii1/nlp-on-activesg-reviews#sentiment-analysis-using-VADER
- JiFacts (Youtube): https://www.youtube.com/watch?v=GVwjR6lkS6Q&list=PL_ATC1as-ksokBELzOjVxt277vh3sno3D&index=1
- Rob Mulla (Youtube): https://www.youtube.com/watch?v=xi0vhXFPegw&list=WL&index=1
