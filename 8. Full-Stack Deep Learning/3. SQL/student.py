import sql_interface as sql

create_student_table = """CREATE TABLE student(
                           name char(20) NOT NULL,
                           surname char(20) NOT NULL,
                           id int NOT NULL PRIMARY KEY,
                           country char(20),
                           FOREIGN KEY(id) REFERENCES projects (st)
                           );
                         """


insert_student = '''INSERT INTO student(name, surname, id) 
                    VALUES('Michal', 'Podlaszuk', 00)'''

select = '''SELECT * FROM student'''



drop_student = '''DROP TABLE student'''

res = sql.sql_query(select)

for item in res:
    print(item)

print(sql.pandas_select(select))