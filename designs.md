# Feature: RESTful API for Retrieving User Activity Logs

---

## 1. Architecture Overview

### 1.1. Component Diagram
graph TD Client[Client Application] API[RESTful API Controller] Service[ActivityLog Service Layer] DB[(Database: ActivityLogs Table)]
Client -->|HTTP GET /api/activitylogs| API
API --> Service
Service --> DB
DB --> Service
Service --> API
API --> Client

- **Client**: Consumes the API to retrieve user activity logs.
- **API Controller**: Exposes endpoints for querying activity logs.
- **Service Layer**: Encapsulates business logic and data access.
- **Database**: Stores activity log records.

---

## 2. Data Flow Description

1. **Request Initiation**:  
   The client sends an HTTP GET request to `/api/activitylogs` with optional query parameters (e.g., `userId`, `fromDate`, `toDate`, `actionType`, `page`, `pageSize`).

2. **API Controller**:  
   The controller receives the request, validates parameters, and delegates to the service layer.

3. **Service Layer**:  
   The service constructs a query based on filters, applies pagination, and retrieves matching records from the `ActivityLogs` table.

4. **Database**:  
   The database executes the query and returns the result set.

5. **Response Construction**:  
   The service maps database entities to DTOs, and the controller returns a paginated JSON response to the client.

---

## 3. API Schema

### 3.1. Endpoint

- **GET** `/api/activitylogs`

### 3.2. Query Parameters

| Name       | Type     | Required | Description                                 |
|------------|----------|----------|---------------------------------------------|
| userId     | string   | No       | Filter logs by user ID                      |
| fromDate   | string   | No       | ISO 8601 date, filter logs from this date   |
| toDate     | string   | No       | ISO 8601 date, filter logs up to this date  |
| actionType | string   | No       | Filter logs by action type                  |
| page       | integer  | No       | Page number (default: 1)                    |
| pageSize   | integer  | No       | Page size (default: 20, max: 100)           |

### 3.3. Response: 200 OK
C->>API: GET /api/activitylogs?userId=123
API->>S: GetActivityLogs(userId=123)
S->>DB: Query ActivityLogs WHERE userId=123
DB-->>S: Result set
S-->>API: List<ActivityLogDto>
API-->>C: 200 OK + JSON response