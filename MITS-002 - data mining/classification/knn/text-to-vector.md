

#### DATA TRANSFORMATION FOR THE EMAIL SPAM/HAM DATASET

---

1. the simple way is via using **Bag of Words** other calls it as **Count Vectorization** eg.

Markings: ``1 = SPAM, 0 = HAM``

Email 1: **Win a free lottery**
Email 2: **Hi, meeting for lunch?**

| Email | free | hi | lottery | lunch | meeting | win |
| -------- | -------- | -------- | -------- | -------- | -------- |-------- |
| **Email 1**   | ``1``   | ``0``   | ``1``   | ``0``   | ``0``   | ``1``   |
| **Email 2**   | ``0``   | ``1``   | ``0``   | ``1``   | ``1``  | ``0``   |

---

2. the advanced way is using **TF-IDF (Term Frequency-Inverse Document)** - is a numerical statistic used in NLP and information retrieval to measure a word's relevance to a document within a collection. It highlights meaningful keywords by multiplying how often a word appears in a document (TF) by the inverse frequency of the word across all documents (IDF).

- using formula breakdown

$$\text{TF}(t,d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$
$$ \text{IDF(T)} = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents with term (T) in it}} \right) $$

- final calculation score

$$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$

Real world example using Email Spam Filters:
- A new email arrives in your inbox.
- Common words like **Hi** or **attached** are ignored. But if the email contains a high frequency of rare words like **lottery,** **inheritance,** or **pharmacy** that don't usually appear in your regular emails, those words get a high TF-IDF score.
- The email is automatically flagged as spam because its **keyword signature** matches known spam patterns
