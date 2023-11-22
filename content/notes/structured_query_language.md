+++
title = "Structured Query Language"
authors = ["Alex Dillhoff"]
date = 2023-10-30T00:00:00-05:00
tags = ["computer science", "databases"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [History and Development](#history-and-development)
- [Schemas](#schemas)
- [Data Types](#data-types)
- [Creation](#creation)
- [Constraints](#constraints)
- [Retrieving Data](#retrieving-data)
- [Modifying Data](#modifying-data)
- [Nested Queries](#nested-queries)
- [Joined Tables](#joined-tables)
- [Aggregate Functions](#aggregate-functions)
- [Grouping](#grouping)
- [`WITH` Clause](#with-clause)
- [Modifying Tables](#modifying-tables)
- [Summary](#summary)

</div>
<!--endtoc-->



## History and Development {#history-and-development}

Structured Query Language (SQL) is a database language for managing data in a relation DBMS. Its original inception was based on a paper by Edgar F. Codd in 1970 titled _A Relational Model of Data for Large Shared Data Banks_ (<a href="#citeproc_bib_item_2">Codd 1970</a>). Two employees working at IBM in the 1970s, Donald D. Chamberlin and Raymond F. Boyce, developed the first version of SQL in 1974 (<a href="#citeproc_bib_item_1">Chamberlin and Boyce 1974</a>).

The first official standard of SQL was SQL-86, or SQL1, which was published in 1986 by the American National Standards Institute (ANSI). The following table shows the release dates of major SQL standards along with a brief description of the changes made in each version.

| Standard | Name | Description                                                                           |
|----------|------|---------------------------------------------------------------------------------------|
| SQL-86   | SQL1 | First official standard of SQL                                                        |
| SQL-89   | SQL2 | Added support for integrity constraints, views, and assertions                        |
| SQL-92   | SQL2 | Added support for triggers, recursive queries, and support for procedural programming |
| SQL:1999 | SQL3 | Added support for object-relational features                                          |
| SQL:2003 | SQL3 | Added support for XML, window functions, and support for regular expressions          |
| SQL:2006 | SQL3 | Added more XML storage features and XQuery support                                    |
| SQL:2008 | SQL3 | Added support for TRUNCATE TABLE and enhanced MERGE statements                        |
| SQL:2011 | SQL3 | Added support for temporal data                                                       |
| SQL:2016 | SQL3 | Added support for JSON                                                                |
| SQL:2023 | SQL3 | Added support for Propery Graph Queries and new JSON features                         |


## Schemas {#schemas}

In our [Introduction to Databases]({{< relref "introduction_to_databases.md" >}}) we discussed the concept of a schema as a definition of the structure of a database. In SQL, a schema is a collection of database objects, such as tables, views, and indexes. A schema is owned by a database user and has the same name as the user. A database user can own multiple schemas, and a schema can be owned by multiple users. A schema can also be owned by a role, which is a collection of users. A role can own multiple schemas, and a schema can be owned by multiple roles.

There are several practical reasons for which we would want to create multiple schemas. For example, a database might be used by both a Human Resources and Healthcare Management application. Creating two separate schemas would ensure that data for each application is kept secure from unauthorized users. Multiple schemas are also used for testing and development processes. Large structural changes to an application may require a new scheme to be created. New features can be developed in the new schema while the old schema is still being used by the application.

The following command creates a new schema named `MedApp` and assigns it to the user `MedAdmin`.

```sql
CREATE SCHEMA MedApp AUTHORIZATION MedAdmin;
```


## Data Types {#data-types}

SQL supports a wide variety of data types. The following table shows the most common data types supported by SQL.

| Data Type      | Description                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------------|
| CHAR(n)        | Fixed-length character string. The maximum length is `n` characters.                                                 |
| VARCHAR(n)     | Variable-length character string. The maximum length is `n` characters.                                              |
| INT            | Integer value. The maximum value is `2^31 - 1`.                                                                      |
| SMALLINT       | Integer value. The maximum value is `2^15 - 1`.                                                                      |
| DECIMAL(i,j)   | Fixed-point number. The maximum precision is `38` digits. The maximum scale is `38` digits.                          |
| NUMERIC(i,j)   | Fixed-point number. The maximum precision is `38` digits. The maximum scale is `38` digits.                          |
| REAL           | Floating-point number. The maximum precision is `6` digits.                                                          |
| DOUBLE         | Floating-point number. The maximum precision is `15` digits.                                                         |
| DATE           | Date value. The range is `1000-01-01` to `9999-12-31`.                                                               |
| TIME           | Time value. The range is `00:00:00` to `23:59:59`.                                                                   |
| TIMESTAMP      | Date and time value. The range is `1000-01-01 00:00:00` to `9999-12-31 23:59:59`.                                    |
| CLOB(n)        | Specifies columns with large text values. Maximum length specified in kilobytes (K), megabytes (M), or gigabytes (G) |
| BIT(n)         | Fixed-length bit string.                                                                                             |
| BIT VARYING(n) | Variable-length bit string.                                                                                          |
| BLOB(n)        | Binary Large Object - used for images, video, and other large items.                                                 |


## Creation {#creation}

Creating schemas, databases, and tables is done with the `CREATE` command. The following command creates a new database named `RPG`.

```sql
CREATE DATABASE RPG;
```

When creating a new table, we must specify the name of the table and the attributes of the table. The following command creates a new table named `Users` with four attributes.

```sql
CREATE TABLE Users (
    user_id INT,
    username VARCHAR(50),
    email VARCHAR(50),
    created_at TIMESTAMP
);
```


## Constraints {#constraints}

Constraints allow us to add rules to our database that ensure the integrity of our data. There are several types of constraints that can be added to a table. For example, if a user is deleted, we may want to delete all of the user's posts as well. This can be accomplished by adding a `CASCADE` constraint to the `DELETE` statement. We can also set a default value to each attribute. Constraints such as `CHECK` and `UNIQUE` can be added to ensure that the data is valid and unique. The following table shows the most common constraints supported by SQL.

| Constraint     | Description                                                                         |
|----------------|-------------------------------------------------------------------------------------|
| NOT NULL       | Ensures that a column cannot have a NULL value.                                     |
| UNIQUE         | Ensures that all values in a column are unique.                                     |
| PRIMARY KEY    | A combination of a NOT NULL and UNIQUE.                                             |
| FOREIGN KEY    | Ensures that values in a column match values in another table's column.             |
| CHECK          | Ensures that all values in a column satisfy a specific condition.                   |
| DEFAULT        | Sets a default value for a column when no value is specified.                       |
| INDEX          | Used to create and retrieve data from the database very quickly.                    |
| AUTO INCREMENT | Automatically generates a unique number when a new record is inserted into a table. |

When creating the `Users` table above, we may want to ensure that the `user_id` attribute is unique. We can do this by adding a `UNIQUE` constraint to the `user_id` attribute. It is also possible to have it auto increment so that we do not have to specify a value for it when inserting a new user.

```sql
CREATE TABLE Users (
    user_id INT UNIQUE AUTO_INCREMENT,
    username VARCHAR(50),
    email VARCHAR(50),
    created_at TIMESTAMP
);
```

The following command creates a new table named `Characters` with a `VARCHAR` attribute named `Name` which is set to `NOT NULL`.

```sql
CREATE TABLE Characters (
    Name VARCHAR(50) NOT NULL
);
```

Constraints can also be added after the initial attribute declaration. When creating the `Characters` table, if we want to state that the `user_id` field should be a foreign key, we can add a `FOREIGN KEY` constraint to the `user_id` attribute.

```sql
CREATE TABLE Characters (
    id INT UNIQUE AUTO_INCREMENT,
    Name VARCHAR(50) NOT NULL,
    user_id INT,
    CONSTRAINT fk_user_id
     FOREIGN KEY (user_id) REFERENCES Users(user_id)
);
```

The constraint is given the name `fk_user_id` and is added to the `user_id` attribute. The `FOREIGN KEY` constraint states that the `user_id` attribute references the `user_id` attribute in the `Users` table.


## Retrieving Data {#retrieving-data}

Retrieving data from an SQL database is done with an `SFW` query, `SELECT-FROM-WHERE`.

```sql
SELECT <attribute list>
FROM <table list>
WHERE <condition>
```

For example, we can get the experience and level of a character named `Atticus` from the `Characters` table with the following query.

```sql
SELECT experience, level
FROM Characters
WHERE Name = 'Atticus';
```

The attributes we retrieve in a query are referred to as the `projection attributes`. This query `SELECT~s a ~Character` from all rows of the `Character` table which satisfy the **selection condition** of the `WHERE` clause.
We can also query the e-mail addresses of all users who have a character that is a human.

```sql
SELECT email
FROM Users, Characters, Races
WHERE Users.user_id = Characters.user_id AND Characters.race_id = Races.id AND Races.name = 'Human';
```

The `WHERE` clause in this example is an example of a **join condition** since it combines attributes from multiple tables. Note that there are two tables which have a `user_id` attribute, so we must differentiate them by prepending the table name before the attribute name. This is how ambiguities are solved in SQL.

You can also use the `AS` keyword to shorthand the table names in your query. The previous query can be rewritten as

```sql
SELECT U.username
FROM   Users AS U, Characters AS C, Races AS R
WHERE  U.user_id = C.user_id AND R.id = C.race_id AND R.name = 'Human';
```


### Duplicate Return Values {#duplicate-return-values}

The previous query returns the names of all users who have a `Human` character. If a user has multiple characters that are `Human`, it will return their name multiple times. If we are instead only interested in the names of users who have a `Human` character, we can use the `DISTINCT` keyword to remove duplicate values.

```sql
SELECT DISTINCT U.username
FROM   Users AS U, Characters AS C, Races AS R
WHERE  U.user_id = C.user_id AND R.id = C.race_id AND R.name = 'Human';
```


### Tables as Sets {#tables-as-sets}

SQL uses some set operations from set theory. It supports the `UNION`, set difference `EXCEPT`, and set intersection `INTERSECT` operations. The following query returns the names of all users who have a `Human` character or a `Gnome` character.

```sql
(SELECT DISTINCT U.username
 FROM Users as U, Characters as C, Races as R
 WHERE U.user_id = C.user_id AND C.race_id = R.id AND R.name = 'Human')
 UNION
(SELECT DISTINCT U.username
 FROM Users as U, Characters as C, Races as R
 WHERE U.user_id = C.user_id AND C.race_id = R.id AND R.name = 'Gnome');
```

If we wanted to find the users who had both a `Human` character and a `Gnome` character, we could use the `INTERSECT` operator instead.

```sql
(SELECT DISTINCT U.username
 FROM Users as U, Characters as C, Races as R
 WHERE U.user_id = C.user_id AND C.race_id = R.id AND R.name = 'Human')
 INTERSECT
(SELECT DISTINCT U.username
 FROM Users as U, Characters as C, Races as R
 WHERE U.user_id = C.user_id AND C.race_id = R.id AND R.name = 'Gnome');
```

We can also use the `EXCEPT` operator to find the users who have a `Human` character but not a `Gnome` character.

```sql
(SELECT DISTINCT U.username
 FROM Users as U, Characters as C, Races as R
 WHERE U.user_id = C.user_id AND C.race_id = R.id AND R.name = 'Human')
 EXCEPT
(SELECT DISTINCT U.username
 FROM Users as U, Characters as C, Races as R
 WHERE U.user_id = C.user_id AND C.race_id = R.id AND R.name = 'Gnome');
```


### Pattern Matching {#pattern-matching}

SQL supports pattern matching with the `LIKE` operator. The `LIKE` operator is used in the `WHERE` clause to search for a specified pattern in a column. This is different from equality operators since it allows us to search for patterns rather than exact matches. The following table shows the most common wildcards used in SQL.

| Wildcard | Description                                           |
|----------|-------------------------------------------------------|
| %        | Matches any string of zero or more characters.        |
| _        | Matches any single character.                         |
| []       | Matches any single character within the brackets.     |
| [^]      | Matches any single character not within the brackets. |

The following query returns the names of all simple items in the `Items` table. These can be found based on their description, since the term `simple` is not explicitly mentioned in the name.

```sql
SELECT  name
FROM    Items
WHERE   description LIKE '%simple%';
```

We can also query based on arithmetic ranges. For example, we might be interested in the items that are less than 100 gold.

```sql
SELECT  name
FROM    Items
WHERE   value < 100;
```


### Ordering {#ordering}

SQL allows us to order the results of our query with the `ORDER BY` clause. The following query returns the names of all items in the `Items` table ordered by their value.

```sql
SELECT  name
FROM    Items
ORDER BY value;
```

We can also order by multiple attributes. The following query returns the names of all items in the `Items` table ordered by their value and then their name.

```sql
SELECT  name
FROM    Items
ORDER BY value, name;
```


## Modifying Data {#modifying-data}


### Inserting Data {#inserting-data}

We previously saw an example of inserting new data. Let's insert a new user account to our table. If we are inserting a value for every attribute, we can omit the attribute list.

```sql
INSERT INTO Users
VALUES (7, 'Alex', 'alex.dillhoff@uta.edu', '2023-10-31 15:26:17');
```

If we are only inserting values for some attributes, we must specify the attribute list.

```sql
INSERT INTO Users (user_id, username, email)
VALUES (7, 'Alex', 'alex.dillhoff@uta.edu');
```

If we attempt to leave out a value for an attribute that is not nullable, we will get an error. While working on our database, we may have realized that some of these important attributes should always be specified. We can add a `NOT NULL` constraint to these attributes to ensure that they are always specified. We will look at ways of modifying tables in the next section.


### Updating Data {#updating-data}

Updating data is a common task and is easily supported by the `UPDATE` command. In an RPG, players will use items, gain experience, and level up. All of these will require modifications to existing tables. For example, if we wish to update the experience of a character, we can use the following query.

```sql
UPDATE Characters
SET experience = experience + 100
WHERE name = 'Atticus';
```


### Deleting Data {#deleting-data}

Deleting a tuple or several tuples is straightforward in SQL. The following query deletes the user with the `user_id` of `7` from the `Users` table.

```sql
DELETE FROM Users
WHERE user_id = 7;
```

If we want to delete all tuples from a table, we can use the `TRUNCATE TABLE` command. This command is faster than deleting all tuples with the `DELETE` command since it does not log each deletion. However, it cannot be used if the table is referenced by a foreign key constraint.

```sql
TRUNCATE TABLE Users;
```

When deleting tuples from a database, it's important to consider any foreign key constraints that the table may have. If we delete a tuple from a table that is referenced by a foreign key constraint, we may end up with orphaned tuples. For example, if we delete a user from the `Users` table, we may end up with a character that has no user. We can avoid this by adding a `CASCADE` constraint to the `DELETE` statement. This will delete all tuples that reference the tuple we are deleting.


## Nested Queries {#nested-queries}

Nested queries allow us to make more complex queries on subsets of data returned from an original query. A nested query can be placed in any of the `SELECT`, `FROM`, or `WHERE` clauses. The query that uses the results of the nested query is called the **outer query**.

What if we wanted a list of users who had at least 1 character whose class was the least represented class across all characters? We could first make a query to identify which class is the least represented before using that to find the users who have a character with that class.

```sql
SELECT  username
FROM    Users
WHERE   user_id IN (SELECT user_id
                    FROM Characters
                    WHERE class_id = (SELECT class_id
                                      FROM Characters
                                      GROUP BY class_id
                                      ORDER BY COUNT(*) ASC
                                      LIMIT 1));
```

The innert-most query returns the `class_id` of the least represented class by counting the number of characters in each class and ordering them in ascending order. The middle query returns the `user_id` of all characters whose class is the least represented class. The outer query returns the `username` of all users who have a character with the least represented class.

Pay attention to the second `WHERE` clause that uses the `=` operator instead of `IN`. This is because the nested query returns a single value and single tuple. If we used the `IN` operator, we would get an error since MySQL does not support the `LIMIT` and `IN/ALL/ANY/SOME` operators together.


### Correlated Nested Queries {#correlated-nested-queries}

Some nested queries would have to execute for each tuple in the outer query. Consider the following query which returns the name and ID of all Human characters.

```sql
SELECT  C.id, C.name
FROM    Characters as C
WHERE   C.race_id IN (SELECT R.id
                      FROM Races as R
                      WHERE R.name = 'Human');
```

This query is an example of a **correlated nested query** since the inner query is dependent on the outer query. The inner query must be executed for each tuple in the outer query. This can be inefficient if the outer query returns a large number of tuples.

Since this query uses only an `IN` operator, we can rewrite it as a single block query.

```sql
SELECT  C.id, C.name
FROM    Characters as C, Races as R
WHERE   C.race_id = R.id AND R.name = 'Human';
```

Let's revisit an earlier query to introduce the `EXISTS` operator. If we want to query the user names and IDs of all users who have an elf character, we can use the following query.

```sql
SELECT  U.id, U.username
FROM    Users as U
WHERE   EXISTS (SELECT *
                FROM   Characters as C, Races as R
                WHERE  C.user_id = U.id AND C.race_id = R.id
                       AND R.name = 'Elf');
```

We can also use the `NOT EXISTS` operator to find the users who do not have an elf character. This works opposite to the `EXISTS` operator.


## Joined Tables {#joined-tables}

Joined tables are the result of a query that combines rows from two or more tables. The syntax itself may be easier to understand as compared to the nested queries written above. The following query returns the names of all users who have a `Human` character.

```sql
SELECT U.username, C.name
FROM   (Users as U JOIN Characters as C ON U.user_id = C.user_id), Races as R
WHERE  C.race_id = R.id AND R.name = 'Human';
```

Here, the `JOIN` function is used to combine both tables on the condition that the `user_id` of the `Users` table is equal to the `user_id` of the `Characters` table. The `WHERE` clause is used to filter the results to only those that have a `Human` character.

This type of join is also referred to as an **inner join** since it only returns rows that satisfy the join condition. There are several other types of joins that we can use to combine tables. If we wanted to return the same information along with all of the users who do not have a `Human` character, we can use a **left outer join**.

```sql
SELECT U.username, C.name
FROM   (Users as U LEFT OUTER JOIN
        (Characters as C JOIN Races as R ON C.race_id = R.id AND R.name = 'Human')
        ON U.user_id = C.user_id);
```


## Aggregate Functions {#aggregate-functions}

Aggregate functions are used to perform calculations on a set of values and return a single value. The following table shows the most common aggregate functions supported by SQL.

| Function | Description                                                 |
|----------|-------------------------------------------------------------|
| AVG()    | Returns the average value of a numeric column.              |
| COUNT()  | Returns the number of rows that match a specified criteria. |
| MAX()    | Returns the maximum value of a column.                      |
| MIN()    | Returns the minimum value of a column.                      |
| SUM()    | Returns the sum of all values in a column.                  |

These can be used in the `SELECT` clause. The following query returns the average level of all characters.

```sql
SELECT AVG(level)
FROM   Characters;
```

These statistics can be used with more complex queries. The following query returns the name of the user with the highest level character.

```sql
SELECT U.username
FROM   Users as U, Characters as C
WHERE  U.user_id = C.user_id AND C.level = (SELECT MAX(level)
                                            FROM Characters);
```

In the next example, the query returns the tuple of the highest level Human character.

```sql
SELECT  C.*
FROM    (Characters as C JOIN Races as R ON C.race_id = R.id AND R.name = 'Human')
WHERE   C.level = (SELECT MAX(level)
                   FROM Characters);
```


## Grouping {#grouping}

As we just saw, aggregate functions permit some preliminary data analysis. We can take this further using **grouping**. For example, we calculated above the average level of all characters. What if we wanted to compute the average level of each class? We can use the `GROUP BY` clause to group the tuples by class.

```sql
SELECT   C.class_id, AVG(C.level)
FROM     Characters as C
GROUP BY C.class_id;
```

In the next example, we will run a similar query except we will also include the number of distinct users who have a character of that class.

```sql
SELECT   C.class_id, AVG(C.level), COUNT(DISTINCT C.user_id)
FROM     Characters as C
GROUP BY C.class_id;
```


## `WITH` Clause {#with-clause}

Let's say we wanted to get the actual class name along with the users that had a character with the least represented class. We can use a nested query in the `FROM` clause to get the class name.

```sql
SELECT  username, Classes.name
FROM    Users, Classes
WHERE   user_id IN (SELECT user_id
                    FROM Characters
                    WHERE class_id = (SELECT class_id
                                      FROM Characters
                                      GROUP BY class_id
                                      ORDER BY COUNT(*) ASC
                                      LIMIT 1))
        AND Classes.id = (SELECT class_id
                          FROM Characters
                          GROUP BY class_id
                          ORDER BY COUNT(*) ASC
                          LIMIT 1);
```

This query might immediately come across as inefficient since we are making the same query multiple times. If you agree, your intuition would be right. We can use the `WITH` clause to make this query more efficient.

```sql
WITH LeastUsedClass AS (
    SELECT class_id
    FROM Characters
    GROUP BY class_id
    ORDER BY COUNT(*) ASC
    LIMIT 1
)
SELECT U.username, C.name
FROM Users U
JOIN Characters CH ON U.user_id = CH.user_id
JOIN Classes C ON C.id = CH.class_id
WHERE CH.class_id = (SELECT class_id FROM LeastUsedClass);
```


## Modifying Tables {#modifying-tables}

Modifying databases used in production is inevitable. Typically, you will modify an offline test version before deploying it, but the process is the same. The `ALTER TABLE` command is used to modify tables. The following table shows the most common modifications that can be made to a table.

| Command | Description                        |
|---------|------------------------------------|
| ADD     | Adds a new column to the table.    |
| DROP    | Deletes a column from the table.   |
| MODIFY  | Changes the data type of a column. |
| RENAME  | Changes the name of a column.      |

When we originally created the `Users` table, we did not specify a `NOT NULL` constraint for the `username` attribute. We can add this constraint with the following command.

```sql
ALTER TABLE Users
MODIFY username VARCHAR(50) NOT NULL;
```

Another mistake that was made was with the name of the ID column. We can rename this column with the following command.

```sql
ALTER TABLE Users
RENAME COLUMN user_id TO id;
```


### Scenario: Updating Foreign Keys {#scenario-updating-foreign-keys}

Let's say we want to delete anything related to a user if that user is deleted from the `Users` table. That means all characters and inventories associated with those characters should be deleted. Currently, attempting to delete a user will fail with the following error:

```bash
SQL Error [1451] [23000]: Cannot delete or update a parent row: a foreign key constraint fails (`rpg`.`Inventory`, CONSTRAINT `Iventory_ibfk_1` FOREIGN KEY (`character_id`) REFERENCES `Characters` (`id`))
```

It looks like the foreign key also has a typo. Let's recreate this foreign key so that it will delete all characters and inventories associated with a user. First we drop the current key.

```sql
ALTER TABLE Inventory
DROP FOREIGN KEY Iventory_ibfk_1;
```

Then we add a new one.

```sql
ALTER TABLE Inventory
ADD CONSTRAINT Inventory_ibfk_1
FOREIGN KEY (character_id) REFERENCES Characters(id) ON DELETE CASCADE;
```


## Summary {#summary}

A typical SQL query consists of a `SELECT` clause, a `FROM` clause, and a `WHERE` clause. The `SELECT` clause specifies the attributes to be returned. The `FROM` clause specifies the tables to be queried. The `WHERE` clause specifies the conditions that must be satisfied for a tuple to be returned.

As we saw in the examples above, there are up to 6 clauses that can be used in a query. The following table shows the clauses that can be used in a query and the order in which they must appear.

| Clause   | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| SELECT   | Specifies the attributes to be returned.                                    |
| FROM     | Specifies the tables to be queried.                                         |
| WHERE    | Specifies the conditions that must be satisfied for a tuple to be returned. |
| GROUP BY | Groups the tuples by a specified attribute.                                 |
| HAVING   | Specifies the conditions that must be satisfied for a group to be returned. |
| ORDER BY | Specifies the order in which the tuples are returned.                       |

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Chamberlin, Donald D., and Raymond F. Boyce. 1974. “SEQUEL: A Structured English Query Language.” In <i>Proceedings of the 1974 ACM SIGFIDET (Now SIGMOD) Workshop on Data Description, Access and Control</i>, 249–64. SIGFIDET ’74. New York, NY, USA: Association for Computing Machinery. <a href="https://doi.org/10.1145/800296.811515">https://doi.org/10.1145/800296.811515</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Codd, E. F. 1970. “A Relational Model of Data for Large Shared Data Banks.” <i>Communications of the Acm</i> 13 (6): 377–87. <a href="https://doi.org/10.1145/362384.362685">https://doi.org/10.1145/362384.362685</a>.</div>
</div>
