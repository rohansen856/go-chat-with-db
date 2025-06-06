# nlp-to-sql

A backend application that enables conversational database interactions, leveraging Retrieval-Augmented Generation (RAG) to generate context-aware, tailored responses. It converts NLP to SQL queries. It takes a textual request and returns a textual response based on the queried data.

#

#### *PROJECT OVERVIEW*

#### Example Question/Request:

- > How many accounts have been opened till date?

#### Generated Query:

```sql
   SELECT COUNT(*) FROM accounts;
```

#### Example Respose:

- > We've got a total of 114 accounts opened so far.

#### API endpoints
1. User Endpoints
- POST /api/v1/user/signup - Create a new user 
> Request body: createUserRequest (username, full_name, email, password)

- POST /api/v1/user/login - Login a user
> Request body: loginUserRequest (email, password)
> Response: loginUserResponse (tokens and user profile)

- PATCH /api/v1/user/update - Update user (authenticated)
> Request body: updateUserRequest (email, username, full_name, password)

- PATCH /api/v1/user/delete - Delete user (authenticated)

2. Admin Endpoints
- POST /api/v1/admin/signup - Create admin user
- POST /api/v1/admin/login - Login admin user
- PATCH /api/v1/admin/update - Update admin (admin authenticated)
- PATCH /api/v1/admin/user/restrict/:userId - Restrict a user (admin authenticated)
- PATCH /api/v1/admin/user/delete/:userId - Delete a user (admin authenticated)

3. WebSocket Endpoints
- GET /api/v1/chat - WebSocket connection for chat (authenticated)

#### *SECURITY CONSIDERATIONS*

- Prompts are engineered to ensure that conversations can only lead to **READ** operations:

  - Conditions in place to ensure that queries generated by the AI model are only `SELECT` statements.
  - Programmatically, generated queries from the AI model are also checked to ensure that only `SELECT` statements are used to query the database.
  - Sensitive data are exempted from the query generated and subsequently from the response provided.

- Database connection strings are not persisted or stored but please ensure that temporary connection strings are created before supplying them during usage. Good to note that they are only used programmatically for establishing database connection and further getting requested data.
