# Can You Judge a Book by Its Cover?

Book covers carry a significant weight in communicating information to potential readers. Herein, we investigate the task of using book cover images and title text to classify the book into one of 30 genres. A 14.8% accuracy was achieved using a CNN model for book covers, and a 44.6% accuracy was achieved by using a multinomial Naive Bayes approach on title text. A linear combination of both models was utilized to achieve a 55.0% accuracy.

The dataset used in this paper has been composed by Iwana et al. and is available at https://github.com/uchidalab/book-dataset. This dataset contains 57,000 books from Amazon according to 30 genres, with each genre having 1900 books. We split the dataset into a 8:1:1 ratio of 45,600 training examples, 5700 validation examples, and 5700 testing examples.


The final model was able to achieve a top 1 accuracy of 55.0% and a top 3 accuracy of 73.8%, which is significantly better than the baseline of 3.3% and 10%, respectively. From this, we see that the cover of a book (the cover image and title) do indeed reveal significant information about the book, particularly the genre.

This project was developed as final project for the CS 229 Course.
