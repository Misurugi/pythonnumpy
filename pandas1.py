import pandas as pd
import numpy as np

#Задание 1
authors = pd.DataFrame({'author_id':[1,2,3],
                        'author_name':['Тургенев', 'Чехов', 'Островский']},
                        columns= ['author_id', 'author_name'])
print(authors)

#Задание 2
book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3],
                      'book_title': ['Отцы и дети', "Рудин", 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                      'price': [450, 300, 350, 500, 450, 370, 290]},
                     columns= ['author_id', 'book_title', 'price'])
print(book)

authors_price = pd.merge(authors, book, on='author_id', how='inner')
print(authors_price)

#Задание 3
top5 = authors_price.nlargest(5, "price")
print(top5)

#Задание 4
max = authors_price.groupby('author_name').agg({'price': 'max'})
min = authors_price.groupby('author_name').agg({'price': 'min'})
mean = authors_price.groupby('author_name').agg({'price': 'mean'})
mean = mean.astype(np.int64)
print(min)
print(max)
print(mean)

author_stat = pd.concat([min, max, mean], axis=1)
author_stat.rename(columns={'price': 'min_price', 'price': 'max_price', 'price': 'mean_price'}, inplace=True)

print(author_stat)