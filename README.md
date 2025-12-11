# NLP-Chatbot
When you're building a chatbot, the first thing you need to do is clean up your text data so the computer can actually understand it. This preprocessing stage has five main steps that work together to transform messy, real-world text into something a machine learning model can work with.

The first step is converting everything to lowercase. This might seem simple, but it's crucial—without it, your chatbot would think "Hello," "hello," and "HELLO" are three completely different words when they all mean the same thing. By standardizing everything to lowercase, you're telling the model to treat these variations as identical.

Next comes tokenization, which is just a fancy way of saying you're breaking text apart into smaller pieces. You can tokenize by sentences (splitting a paragraph into individual sentences) or by words (breaking sentences into individual words). This gives you manageable chunks that the computer can analyze one piece at a time instead of trying to process huge blocks of text all at once.

The third step is noise removal, where you strip out everything that isn't a letter or number. Think of all those special characters, punctuation marks, and random symbols people use when typing—most of that stuff doesn't add meaning for a chatbot, it just clutters up the data. By removing it, you're left with clean, pure text content.

Step four is removing stopwords, which are extremely common words like "the," "is," "and," "in," etc. These words appear constantly but don't carry much meaningful information. Removing them helps your model focus on the words that actually matter. However, you have to be careful here—words like "not" and "no" can completely flip the meaning of a sentence, so sometimes you want to keep certain stopwords intact.

Finally, there's lemmatization, which reduces words to their root form. For example, "running," "runs," and "ran" all get converted back to the base word "run." This helps your chatbot recognize that these are all variations of the same concept rather than treating them as completely separate words. It's similar to stemming but more sophisticated because it actually understands word meanings and grammar rather than just chopping off endings.

All five of these steps work together in your preprocessing pipeline, and you can see them implemented in the preprocessing.py file. When I built your chatbots, both of them use these exact techniques to clean and prepare text before doing any analysis or classification. The slides from weeks 5 and 6 have more details if you want to dive deeper into the theory behind each step.
