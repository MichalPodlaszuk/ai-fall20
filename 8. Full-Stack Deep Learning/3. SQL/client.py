import sql_interface as sql

email = 'michalpodlaszu@gmail.com'
while True:
    sentence = input('What is your query? \n')
    if sentence.split()[0].lower() == 'select':
        res = sql.pandas_select(sentence)
        print(email not in list(res))
    else:
        print(sql.sql_query(sentence))