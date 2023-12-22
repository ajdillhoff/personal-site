+++
title = "Introduction to Databases"
authors = ["Alex Dillhoff"]
date = 2023-10-28T00:00:00-05:00
tags = ["computer science", "databases"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [An Online RPG Database](#an-online-rpg-database)
- [From Schema to Database](#from-schema-to-database)
- [Database Management Systems](#database-management-systems)
- [Creating our RPG Database](#creating-our-rpg-database)

</div>
<!--endtoc-->

**Recommended Reading: Chapters 1 and 2 from (<a href="#citeproc_bib_item_1">Elmasri and Navathe 2015</a>)**

Databases allow us to store, retrieve, and edit different types of data. They should be scalable, secure, and reliable. They should also be able to handle concurrent access and be able to recover from failures. There are multiple types of databases that are optimized for different use cases. Tabular data, for example, is typically stored in a **relational** database. Large format data such as images, videos, and audio are typically stored in a **non-relational** database.

Creating, deploying, and maintaining databases is facilitated through a **database management system** (DBMS). A DBMS is a software system that allows us to interact with a database. It provides an interface for us to create, read, update, and delete data. It also provides a way for us to define the structure of our data and the relationships between different pieces of data. Examples of DBMSs include MySQL, PostgreSQL, and MongoDB.

Once a database is deployed, we can interact with it a number of ways. Most DBMSs include a client which allows us to interact with the database through a command line interface. We can also interact with the database through a programming language such as Python or Java.

It is important to emphasize that a database is not the same thing as a file system. A file system is a way to store data on a disk, whereas a database is a way to store data in a file system. File systems are good at managing unstructured data with little regard to the relationships inherit in the data itself. What if multiple people working on the same document try to save their changes at the same time? What if a user tries to delete a file that is currently being used by another user? These are problems that a file system is not designed to handle.


## An Online RPG Database {#an-online-rpg-database}

To introduce some foundational terms and concepts of databases, let's design and create a database for an online RPG. In this game, users can create accounts, make multiple characters, store items for their characters, and embark on quests to level up their characters. Even from this simple description, we can start separating our data into different **entities** and **relationships**. Each logical entity in our game will be represented by a **table** in our database. The **attributes** of each table will be represented by columns in our database. For this database, we will need _at least_ the following tables:

-   `Users`
-   `Characters`
-   `Items`
-   `Inventory`
-   `Quests`

We may add or modify these depending on the finer details. If you are not familiar with online RPG games, don't worry. We will be sure to include the necessities to get us started. Let's start with the first table, `Users`.


### Users {#users}

A `User` represents an online account that is unique to each person who plays the game. It should contain the `username`, `email`, and date that it was created, which we will call `created_at`. This is enough information for now. Using this, we can create our first **table**. There is one more attribute that wasn't explicitly mentioned. Each `User` in our table should have a unique identifier. This is called a **primary key**. We will use a sequentially increasing number starting at 1 for our primary key. This is a common practice, but it is not the only way to do it. We will call this column `user_id`. The full table is showing below.

**Users**

-   `user_id`: primary key
-   `username`
-   `email`
-   `created_at`


### Characters {#characters}

It is common for users to have multiple characters so they can experience the full range of our game. This table will have more attributes than the `Users` table since there are a wide range of stats that our characters can have, such as their name, level, experience, and health. We will also need to know which user each character belongs to. We can do this by adding a column called `user_id` which will be a **foreign key** to the `Users` table. This will allow us to link each character to the user that created it. The full table is shown below.

**Characters**

-   `character_id`: primary key
-   `user_id`: foreign key
-   `name`
-   `level`
-   `experience`
-   `health`
-   `created_at`


### Items {#items}

As our user's play, they will collect items such as weapons, armor, and potions. As our game evolves, our game designers will add more items to the game. A table for our items is shown below.

**Items**

-   `item_id`: primary key
-   `name`
-   `value`


### Inventory {#inventory}

Our users will need a way to store their items. We can do this by creating a table called `Inventory`. This table will have a foreign key to the `Characters` table so we can link each item to the character that owns it. It will also have a foreign key to the `Items` table so we can link each item to the item that it represents. We will also need to know how many of each item our users have. We can do this by adding a column called `quantity`. The full table is shown below.

**Inventory**

-   `inventory_id`: primary key
-   `character_id`: foreign key
-   `item_id`: foreign key
-   `quantity`


### Quests {#quests}

No RPG would be complete without quests that our player's could embark upon. The `Quests` table will have a name, description, and a reward. In the case of multiple rewards, we can create a separate table called `QuestRewards` that will have a foreign key to the `Quests` table and a foreign key to the `Items` table. This will allow us to link each quest to the items that it rewards. This means that the `Quests` table does not need an explicit reference to the reward item. We can look those up separately. The full table is shown below.

**Quests**

-   `quest_id`: primary key
-   `name`
-   `description`
-   `reward_experience`
-   `min_level`

**QuestRewards**

-   `quest_reward_id`: primary key
-   `quest_id`: foreign key
-   `item_id`: foreign key


### A Few Extras {#a-few-extras}

There are a few more tables we should add to round out our characters. Most RPGs allow the users to create characters of different **races**, such as a human, orc, or elf, as well as the characters **class**, which defines what sort of abilities the character will have.

**Race**

-   `race_id`: primary key
-   `name`

**Class**

-   `class_id`: primary key
-   `name`

With the addition of these two tables, let's add _foreign keys_ to our original `Characters` table. We will add a `race_id` and a `class_id`. The full table is shown below.

**Characters**

-   `character_id`: primary key
-   `user_id`: foreign key
-   `name`
-   `level`
-   `experience`
-   `health`
-   `race_id`: foreign key
-   `class_id`: foreign key
-   `created_at`

That's it! We have all the tables we need to get us started. All tables with example data are shown below. You'll notice that each of the primary IDs in the tables below have been renamed to `id`. Besides giving us extra room to display the table, the primary key is always unique to the table, so we don't need to include the table name in the column name.

**Users**

\begin{array}{|r|l|l|l|}
\hline
\text{id} & \text{username} & \text{email} & \text{created\_at} \\\\
\hline
1 & \text{Naomi} & \text{player1@example.com} & \text{2023-01-01 10:00:00} \\\\
2 & \text{Clarissa} & \text{player2@example.com} & \text{2023-01-02 11:00:00} \\\\
3 & \text{Avasarala} & \text{player3@example.com} & \text{2023-01-03 12:00:00} \\\\
\hline
\end{array}

**Characters**

\begin{array}{|r|r|l|r|r|r|r|r|l|}
\hline
\text{id} & \text{user\_id} & \text{name} & \text{class\_id} & \text{race\_id} & \text{level} & \text{experience} & \text{health} & \text{created\_at} \\\\
\hline
1 & 1 & \text{Atticus} & 1 & 1 & 10 & 1000 & 100 & \text{2023-01-01 10:10:00} \\\\
2 & 1 & \text{Bobbie} & 2 & 2 & 15 & 1500 & 200 & \text{2023-01-01 10:20:00} \\\\
3 & 2 & \text{Raimi} & 3 & 3 & 8 & 800 & 90 & \text{2023-01-02 11:10:00} \\\\
4 & 3 & \text{Beef} & 4 & 4 & 12 & 1200 & 110 & \text{2023-01-03 12:10:00} \\\\
5 & 2 & \text{Demon} & 4 & 4 & 12 & 1200 & 110 & \text{2023-01-05 12:10:00} \\\\
\hline
\end{array}

**Items**

\begin{array}{|r|l|r|r|}
\hline
\text{id} & \text{name} & \text{value} \\\\
\hline
1 & \text{Sword} & 100 \\\\
2 & \text{Shield} & 150 \\\\
3 & \text{Staff} & 200 \\\\
4 & \text{Bow} & 250 \\\\
\hline
\end{array}

**Inventory**

\begin{array}{|r|r|r|r|}
\hline
\text{id} & \text{character\_id} & \text{item\_id} & \text{quantity} \\\\
\hline
1 & 1 & 1 & 1 \\\\
2 & 2 & 2 & 1 \\\\
3 & 3 & 3 & 1 \\\\
4 & 4 & 4 & 1 \\\\
\hline
\end{array}

**Quests**

\begin{array}{|r|l|l|r|l|r|}
\hline
\text{id} & \text{name} & \text{description} & \text{reward\_experience} & \text{min\_level} \\\\
\hline
1 & \text{Linken's Sword} & \text{Desc1} & 100 & 5 \\\\
2 & \text{Mankrik's Wife} & \text{Desc2} & 200 & 10 \\\\
3 & \text{The Hermit} & \text{Desc3} & 300 & 15 \\\\
4 & \text{The Great Masquerade} & \text{Desc4} & 400 & 20 \\\\
\hline
\end{array}

**QuestRewards**

\begin{array}{|r|r|r|}
\hline
\text{id} & \text{quest\_id} & \text{item\_id} \\\\
\hline
1 & 1 & 1 \\\\
2 & 2 & 2 \\\\
3 & 3 & 3 \\\\
4 & 4 & 4 \\\\
\hline
\end{array}

**Races**

\begin{array}{|r|l|}
\hline
\text{race\_id} & \text{name} \\\\
\hline
1 & \text{Human} \\\\
2 & \text{Elf} \\\\
3 & \text{Dwarf} \\\\
4 & \text{Orc} \\\\
\hline
\end{array}

**Classes**

\begin{array}{|r|l|}
\hline
\text{class\_id} & \text{name} \\\\
\hline
1 & \text{Warrior} \\\\
2 & \text{Mage} \\\\
3 & \text{Rogue} \\\\
4 & \text{Paladin} \\\\
\hline
\end{array}


## From Schema to Database {#from-schema-to-database}

What we did in the previous example is created a database **schema** based on our entities. A schema does not represent the entire picture of our **data model**. Relationships and other constraints are not represented in the schema. The data model itself defines the structure of a database, including data types, relationships, constraints, and a set of operations for performing basic functions like retrieving and updating data.


### The Three-Schema Architecture {#the-three-schema-architecture}

The three-schema architecture is a way to separate the different aspects of a database. The three schemas are the **external schema**, the **conceptual schema**, and the **internal schema**. The internal schema describes how the data is stored on disk. Unless we are working on the backend of the database, we typically do not need to worry about the internal level. The external schema describes how the data is viewed by the user. This is the level that we interact with when we use a DBMS. The conceptual schema is the middle layer that describes the logical structure of the data. This is the level that we are working with when we create a schema.

Under this architecture, we can modify the internal schema without affecting the external schema. This is important because it allows us to change the way that the data is stored without affecting the applications that use it. We can also modify the external schema without affecting the internal schema. This allows us to change the way that the data is viewed without affecting the applications that use it. This concept of **data independence** is one of the most important features of a DBMS.


## Database Management Systems {#database-management-systems}

With our database defined, we can use it to make **queries** about the records that it stores. How we access that database depends on the DBMS that we are using. The database itself is can be modified and changed without affecting the applications that use it. We can also create multiple **views** of our data dynamically. For example, we can create a view that shows all of the items that a user has in their inventory, or show all of the characters that belong to a specific user. This is all done without modifying the underlying data. This is a powerful feature of databases that allows us to create complex applications that can be easily modified and updated.

A **transaction** is a set of operations that are performed on a database. Transactions are typically used to ensure that the database is in a consistent state. For example, if we want to transfer money from one account to another, we need to make sure that the money is removed from one account and added to the other. If we fail to do this, we could end up with money that is neither in the original account nor the destination account. Transactions allow us to perform these operations in a way that guarantees that the database is in a consistent state.

A DBMS must ensure transactional properties such as **isolation**, which ensure that each transaction executes in isolation from others, and **atomicity**, which ensures that either all operations in a transaction are executed or none are.


### DBMS Languages {#dbms-languages}

A DBMS provides a way for us to interact with the database. Depending on the level of abstraction and the DBMS itself, a specific language is used to perform basic operations on the database. The most common languages are **data definition languages** (DDLs) and **data manipulation languages** (DMLs). A DDL is used to define the structure of the database, such as creating tables and defining relationships between them. A DML is used to perform operations on the data itself, such as inserting, updating, and deleting records.

A common query language called Structured Query Language (SQL) defines both DDLs and DMLs. For example, to create our `User` table from above, we can use the following SQL statement:

```sql
CREATE TABLE Users (
  user_id INT PRIMARY KEY,
  username VARCHAR(255),
  email VARCHAR(255),
  created_at DATETIME
);
```

Note that we must specify a type for each attribute in our table. SQL also provides a DML, we can use to insert records into our table:

```sql
INSERT INTO Users (user_id, username, email, created_at)
VALUES (1, 'Naomi', 'player1@example.com', '2023-01-01 10:00:00');
```


### DBMS Interfaces {#dbms-interfaces}

A DBMS provides an interface for us to interact with the database. This interface can be a command line interface, a graphical user interface, or a programming language interface. Other interfaces using natural language or voice can also be found in the wild. With the rapid advancement of machine learning, these interfaces are becoming more and more common. [Here](https://github.com/kulltc/chatgpt-sql) is an example of a chatbot that can be used to query a database.


## Creating our RPG Database {#creating-our-rpg-database}

For this example, we will be using MySQL. We only want to make sure that we have MySQL installed and are able to interface with the command line. You can find a thorough installation guide [here](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/). Once it is installed and configured, start the MySQL server and log in using the following command:

```bash
mysql -u root -p
```

You should be prompted for a password. If you have not set a password, you can leave it blank. Once you are logged in, you should see a prompt that looks like this:

```bash
mysql>
```

Let's create a database for our RPG. We can do this with the following command:

```sql
CREATE DATABASE rpg;
```

We can verify that the database was created by listing all of the databases on the server:

```sql
SHOW DATABASES;
```

You should see the `rpg` database in the list. We can now use this database to create our tables. We can do this with the following command:

```sql
USE rpg;
```

This will tell MySQL to use the `rpg` database for all subsequent commands. We can now create our `Users` table:

```sql
CREATE TABLE Users (
  user_id INT PRIMARY KEY,
  username VARCHAR(255),
  email VARCHAR(255),
  created_at DATETIME
);
```

We can verify that the table was created by listing all of the tables in the database:

```sql
SHOW TABLES;
```

You should see the `Users` table in the list. We can now insert some data into the table:

```sql
INSERT INTO Users (user_id, username, email, created_at)
VALUES
         (1, 'Naomi', 'player1@example.com', '2023-01-01 10:00:00'),
         (2, 'Clarissa', 'player2@example.com', '2023-01-02 11:00:00'),
         (3, 'Avasarala', 'player3@example.com', '2023-01-03 12:00:00');
```

We can verify that the data was inserted by querying the table:

```sql
SELECT * FROM Users;
```

You should see the data that we inserted in the table. We can now create the rest of our tables:

```sql
CREATE TABLE Characters (
  character_id INT PRIMARY KEY,
  user_id INT,
  name VARCHAR(255),
  level INT,
  experience INT,
  health INT,
  created_at DATETIME
);

CREATE TABLE Items (
  item_id INT PRIMARY KEY,
  name VARCHAR(255),
  value INT
);

CREATE TABLE Inventory (
  inventory_id INT PRIMARY KEY,
  character_id INT,
  item_id INT,
  quantity INT
);

CREATE TABLE Quests (
  quest_id INT PRIMARY KEY,
  name VARCHAR(255),
  description VARCHAR(255),
  reward_experience INT,
  min_level INT
);

CREATE TABLE QuestRewards (
  quest_reward_id INT PRIMARY KEY,
  quest_id INT,
  item_id INT
);
```

Try creating the tables for the `Races` and `Classes` yourself. Once you are done, you can insert some data into the tables. Use the samples from above or create your own. Once you are done, you can query the tables to verify that the data was inserted correctly.

## References

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Elmasri, Ramez, and Shamkant B. Navathe. 2015. <i>Fundamentals of Database Systems</i>. 7th ed. Pearson. <a href="https://www.pearson.com/en-us/subject-catalog/p/fundamentals-of-database-systems/P200000003546/9780137502523">https://www.pearson.com/en-us/subject-catalog/p/fundamentals-of-database-systems/P200000003546/9780137502523</a>.</div>
</div>
