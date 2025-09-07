# Software Oriented Document: QLoRA Orchestration Service

## 1. Introduction
This document provides a comprehensive overview of the software architecture, design, and implementation considerations for the QLoRA Orchestration Service. It covers the System Requirements Specification (SRS), System Design Specification (SDS), and conceptual implementation diagrams, aiming to provide an industry-grade blueprint for development.

## 2. System Requirements Specification (SRS)

### 2.1. Functional Requirements

#### FR1: User Management
-   FR1.1: The system shall allow users to register with a unique username, email, and password.
-   FR1.2: The system shall allow registered users to log in using their credentials.
-   FR1.3: The system shall authenticate users and issue secure access tokens (JWT).

#### FR2: Dataset Management
-   FR2.1: The system shall allow authenticated users to upload new video datasets.
-   FR2.2: The system shall allow authenticated users to register datasets via URL or cloud storage paths.
-   FR2.3: The system shall list all datasets accessible by the user.
-   FR2.4: The system shall allow users to view detailed information about a specific dataset.
-   FR2.5: The system shall allow users to delete their datasets.

#### FR3: Model Management
-   FR3.1: The system shall allow authenticated users to register new QLoRA-compatible video models.
-   FR3.2: The system shall list all registered models accessible by the user.
-   FR3.3: The system shall allow users to view detailed information about a specific model.
-   FR3.4: The system shall allow users to delete their registered models.

#### FR4: Training Orchestration
-   FR4.1: The system shall allow authenticated users to initiate a new QLoRA fine-tuning job by selecting a registered model and dataset.
-   FR4.2: The system shall allow users to specify QLoRA-specific parameters (e.g., rank, alpha, dropout) and training parameters (e.g., epochs, learning rate, batch size).
-   FR4.3: The system shall provide real-time status updates for ongoing training jobs.
-   FR4.4: The system shall allow users to view logs and metrics for completed or ongoing training jobs.
-   FR4.5: The system shall allow users to stop an active training job.
-   FR4.6: The system shall provide an API endpoint for retrieving the fine-tuned model weights or an inference endpoint upon job completion.

#### FR5: Monitoring and Logging
-   FR5.1: The system shall log all significant events, including user actions, training job status changes, and errors.
-   FR5.2: The system shall provide mechanisms to view training progress and resource utilization.

### 2.2. Non-Functional Requirements

#### NFR1: Performance
-   NFR1.1: The API response time for authentication and management operations shall be under 500ms for 95% of requests.
-   NFR1.2: Training job initiation shall be acknowledged within 2 seconds.
-   NFR1.3: The system shall efficiently manage GPU memory during training processes, leveraging QLoRA's benefits.

#### NFR2: Security
-   NFR2.1: User passwords shall be securely hashed and stored.
-   NFR2.2: All API communication shall be secured using HTTPS.
-   NFR2.3: Access to resources shall be restricted based on user authentication and authorization.

#### NFR3: Scalability
-   NFR3.1: The system shall be designed to scale horizontally to accommodate an increasing number of concurrent users and training jobs.
-   NFR3.2: The training backend shall support distributed training environments (conceptual, for future).

#### NFR4: Usability
-   NFR4.1: The frontend UI shall be intuitive and easy to navigate for users with varying technical expertise.
-   NFR4.2: The UI shall provide clear feedback on user actions and system status.

#### NFR5: Maintainability
-   NFR5.1: The codebase shall be modular, well-documented, and adhere to established coding standards.
-   NFR5.2: Dependencies shall be managed effectively to facilitate updates and reduce conflicts.

#### NFR6: Reliability
-   NFR6.1: The system shall handle errors gracefully and provide informative messages.
-   NFR6.2: Training jobs shall be resilient to transient failures and support retry mechanisms where appropriate.

## 3. System Design Specification (SDS)

### 3.1. Architectural Overview
The QLoRA Orchestration Service will adopt a microservices-oriented architecture, separating concerns into distinct, independently deployable services. This design promotes scalability, fault isolation, and technology flexibility. The primary components include a Frontend UI, a Backend API Gateway, and a Training Orchestration Service.

```mermaid
C4Context
    title System Context for QLoRA Orchestration Service

    Person(user, "User", "A user who interacts with the QLoRA Orchestration Service to manage datasets, models, and training jobs.")

    System(frontend_ui, "Frontend UI", "The web-based user interface for interacting with the QLoRA Orchestration Service. Built with Next.js, Tailwind CSS, Shadcn/ui, Aceternity UI, TypeScript.")
    System(backend_api_gateway, "Backend API Gateway", "The central entry point for all API requests, routing them to appropriate microservices. Built with FastAPI.")
    System(auth_service, "Authentication Service", "Handles user registration, login, and token management. Built with FastAPI.")
    System(db_service, "Database Service", "Manages persistent storage for application data (users, datasets, models, training jobs). Uses PostgreSQL.")
    System(training_orchestration_service, "Training Orchestration Service", "Manages the lifecycle of training jobs, including queuing, scheduling, and monitoring. Built with FastAPI, Celery/RabbitMQ.")
    System(training_workers, "Training Workers/Containers", "Executes the actual QLoRA fine-tuning processes. Runs on Jetson Nano emulators or VMs. Uses Python, PyTorch, bitsandbytes, PEFT.")
    System(model_dataset_storage, "Model/Dataset Storage", "Stores raw video datasets, pre-trained model weights, and fine-tuned QLoRA adapter weights. S3-compatible object storage.")
    System(monitoring_logging_service, "Monitoring/Logging Service", "Aggregates logs and metrics from all services and training workers. Uses Prometheus/Grafana, ELK Stack.")

    Rel(user, frontend_ui, "Uses", "HTTPS")
    Rel(frontend_ui, backend_api_gateway, "Makes API calls to", "HTTPS")
    Rel(backend_api_gateway, auth_service, "Authenticates users via")
    Rel(backend_api_gateway, db_service, "Manages data via")
    Rel(backend_api_gateway, training_orchestration_service, "Initiates and queries training jobs via")
    Rel(training_orchestration_service, db_service, "Manages job records in")
    Rel(training_orchestration_service, training_workers, "Dispatches and monitors training tasks on")
    Rel(training_workers, model_dataset_storage, "Accesses and stores data in")
    Rel(training_workers, monitoring_logging_service, "Sends logs and metrics to")
    Rel(db_service, model_dataset_storage, "Stores metadata for")
    Rel(monitoring_logging_service, frontend_ui, "Provides dashboards for (indirectly)")

```

### 3.2. Component Breakdown

#### 3.2.1. Frontend UI
-   **Technology Stack:** Next.js, Tailwind CSS, Shadcn/ui, Aceternity UI, TypeScript.
-   **Responsibilities:** User interface rendering, user interaction handling, API request initiation, client-side validation.

#### 3.2.2. Backend API Gateway
-   **Technology Stack:** FastAPI (Python).
-   **Responsibilities:** Exposing RESTful API endpoints, request routing, input validation, authentication/authorization enforcement, communication with backend services.

#### 3.2.3. Authentication Service
-   **Technology Stack:** FastAPI (Python), PostgreSQL/MongoDB (for user data).
-   **Responsibilities:** User registration, login, token generation (JWT), password hashing.

#### 3.2.4. Database Service
-   **Technology Stack:** PostgreSQL (for structured data like user profiles, dataset metadata, model metadata, training job records).
-   **Responsibilities:** Persistent storage and retrieval of application data.

#### 3.2.5. Training Orchestration Service
-   **Technology Stack:** FastAPI (Python), Celery/RabbitMQ (for job queueing), Docker/Kubernetes (for container orchestration).
-   **Responsibilities:** Receiving training requests, queuing jobs, provisioning and managing training workers (e.g., on Jetson Nano emulators or VMs), monitoring job progress, collecting results.

#### 3.2.6. Training Workers/Containers
-   **Technology Stack:** Python, PyTorch, `bitsandbytes`, `peft`, specific video-text model libraries.
-   **Responsibilities:** Executing the actual QLoRA fine-tuning process on provisioned hardware (e.g., NVIDIA Jetson Nano container emulator), accessing datasets and models, logging metrics.

#### 3.2.7. Model/Dataset Storage
-   **Technology Stack:** S3-compatible object storage (e.g., MinIO, AWS S3) or shared network file system.
-   **Responsibilities:** Storing raw video datasets, pre-trained model weights, and fine-tuned QLoRA adapter weights.

#### 3.2.8. Monitoring/Logging Service
-   **Technology Stack:** Prometheus/Grafana (for metrics), ELK Stack (Elasticsearch, Logstash, Kibana) or Loki/Promtail (for logs).
-   **Responsibilities:** Aggregating logs and metrics from all services and training workers, providing dashboards for system health and training progress.

### 3.3. Data Flow (Example: Initiating a Training Job)

1.  User interacts with Frontend UI to configure a new training job.
2.  Frontend sends a `POST /trainings` request to the Backend API Gateway.
3.  API Gateway validates the request, authenticates the user, and forwards it to the Training Orchestration Service.
4.  Training Orchestration Service validates input, creates a new job record in the Database Service, and adds the job to the Job Queue.
5.  A free Training Worker picks up the job from the Job Queue.
6.  Training Worker pulls the specified dataset and model from Model/Dataset Storage.
7.  Training Worker executes the QLoRA fine-tuning process.
8.  During training, the worker sends progress updates and logs to the Monitoring/Logging Service.
9.  Upon completion, the worker saves the fine-tuned QLoRA adapters to Model/Dataset Storage and updates the job status in the Database Service.
10. Frontend UI periodically polls the API Gateway (`GET /trainings/{job_id}`) to update the user on job status and progress.

## 4. Module Design and Implementation (Conceptual)

### 4.1. User Module
-   **Entities:** User (username, email, hashed_password, roles).
-   **API Endpoints:** `/auth/login`, `/auth/signup`, `/users/me`.
-   **Key Logic:** Password hashing (Bcrypt), JWT generation and validation.

### 4.2. Dataset Module
-   **Entities:** Dataset (id, name, description, source_type, source_path, status, owner_id).
-   **API Endpoints:** `/datasets`, `/datasets/{id}`.
-   **Key Logic:** File upload handling, metadata storage, status tracking (e.g., `processing`, `ready`, `error`).

### 4.3. Model Module
-   **Entities:** Model (id, name, description, model_type, base_architecture, source_path, status, owner_id).
-   **API Endpoints:** `/models`, `/models/{id}`.
-   **Key Logic:** Model registration, metadata storage.

### 4.4. Training Module
-   **Entities:** TrainingJob (id, model_id, dataset_id, params, status, progress, owner_id, logs_path, results_path).
-   **API Endpoints:** `/trainings`, `/trainings/{id}`, `/trainings/{id}/stop`.
-   **Key Logic:** Job queuing, worker assignment, status updates, result storage.

### 4.5. Core QLoRA Fine-tuning Logic (within Training Worker)
-   **Components:**
    -   `model_loader.py`: Loads base video-text models.
    -   `qlora_injector.py`: Applies 4-bit quantization and injects LoRA adapters.
    -   `data_processor.py`: Handles video frame sampling, text tokenization, and batching.
    -   `trainer.py`: Implements the fine-tuning loop, including optimizer, loss function, and metric calculation.
    -   `temporal_aware_lora.py` (Advanced): Custom LoRA modules for temporal dynamics.
    -   `cross_modal_rank_allocator.py` (Advanced): Logic for dynamic rank assignment.

## 5. UML Diagrams

### 5.1. Use Case Diagram
```mermaid
UMLet
Actor(User)
UseCase(Register User)
UseCase(Login)
UseCase(Manage Datasets)
UseCase(Manage Models)
UseCase(Initiate Training Job)
UseCase(Monitor Training Job)
UseCase(Stop Training Job)

User -- (Register User)
User -- (Login)
User -- (Manage Datasets)
User -- (Manage Models)
User -- (Initiate Training Job)
User -- (Monitor Training Job)
User -- (Stop Training Job)

(Manage Datasets) <. (Add Dataset) : <<include>>
(Manage Datasets) <. (View Dataset Details) : <<include>>
(Manage Datasets) <. (Delete Dataset) : <<include>>

(Manage Models) <. (Add Model) : <<include>>
(Manage Models) <. (View Model Details) : <<include>>
(Manage Models) <. (Delete Model) : <<include>>

(Initiate Training Job) <. (Select Dataset) : <<include>>
(Initiate Training Job) <. (Select Model) : <<include>>
(Initiate Training Job) <. (Configure Training Parameters) : <<include>>

```

### 5.2. Class Diagram
```mermaid
classDiagram
    class User {
        +String id
        +String username
        +String email
        +String passwordHash
        +List~String~ roles
        +register()
        +login()
    }

    class Dataset {
        +String id
        +String name
        +String description
        +String sourceType
        +String sourcePath
        +String status
        +String ownerId
        +DateTime createdAt
        +upload()
        +register()
        +delete()
    }

    class Model {
        +String id
        +String name
        +String description
        +String modelType
        +String baseArchitecture
        +String sourcePath
        +String status
        +String ownerId
        +DateTime createdAt
        +register()
        +delete()
    }

    class TrainingJob {
        +String id
        +String modelId
        +String datasetId
        +Map~String, Any~ trainingParams
        +String status
        +Float progress
        +String ownerId
        +DateTime startedAt
        +DateTime completedAt
        +String logsPath
        +String resultsPath
        +initiate()
        +updateStatus()
        +stop()
    }

    User "1" -- "*" Dataset : owns
    User "1" -- "*" Model : owns
    User "1" -- "*" TrainingJob : initiates

    TrainingJob "1" -- "1" Model : uses
    TrainingJob "1" -- "1" Dataset : uses

    class BackendAPIGateway {
        +authenticateUser()
        +manageDatasets()
        +manageModels()
        +orchestrateTraining()
    }

    class TrainingOrchestrationService {
        +queueJob()
        +monitorJob()
        +dispatchJob()
    }

    class TrainingWorker {
        +executeTraining()
        +reportStatus()
    }

    BackendAPIGateway ..> User : authenticates
    BackendAPIGateway ..> Dataset : manages
    BackendAPIGateway ..> Model : manages
    BackendAPIGateway ..> TrainingJob : orchestrates

    TrainingOrchestrationService "1" -- "*" TrainingJob : manages
    TrainingOrchestrationService "1" -- "*" TrainingWorker : dispatches to

```

### 5.3. Sequence Diagram: Initiate Training Job
```mermaid
sequenceDiagram
    actor User
    participant FrontendUI as Frontend UI
    participant APIGateway as Backend API Gateway
    participant AuthSvc as Authentication Service
    participant TOS as Training Orchestration Service
    participant DBSvc as Database Service
    participant JobQueue as Job Queue
    participant TrainingWorker as Training Worker

    User->>FrontendUI: Clicks "Initiate Training" button
    FrontendUI->>FrontendUI: Gathers training parameters (model, dataset, etc.)
    FrontendUI->>APIGateway: POST /trainings (Auth Token, Job Details)
    APIGateway->>AuthSvc: Validate Token
    AuthSvc-->>APIGateway: Token Valid / Invalid
    APIGateway->>TOS: Forward Training Request
    TOS->>DBSvc: Create new TrainingJob record (status: queued)
    DBSvc-->>TOS: TrainingJob ID
    TOS->>JobQueue: Add TrainingJob ID to queue
    JobQueue-->>TOS: Job added confirmation
    TOS-->>APIGateway: 202 Accepted (Job ID, Status: queued)
    APIGateway-->>FrontendUI: 202 Accepted (Job ID, Status: queued)
    FrontendUI->>User: Displays "Training job initiated" message

    loop Job Monitoring
        FrontendUI->>APIGateway: GET /trainings/{job_id}
        APIGateway->>TOS: Query Job Status
        TOS->>DBSvc: Retrieve Job Status
        DBSvc-->>TOS: Job Status (e.g., running, progress)
        TOS-->>APIGateway: Job Status
        APIGateway-->>FrontendUI: Job Status
        FrontendUI->>User: Updates UI with progress
    end

    TrainingWorker->>JobQueue: Polls for new jobs
    JobQueue-->>TrainingWorker: TrainingJob ID
    TrainingWorker->>DBSvc: Update TrainingJob status (status: running)
    TrainingWorker->>TrainingWorker: Execute QLoRA Fine-tuning
    TrainingWorker->>TrainingWorker: (Accesses Model/Dataset Storage, Logs Metrics)
    TrainingWorker->>DBSvc: Update TrainingJob status (status: completed/failed, results_path)
    DBSvc-->>TrainingWorker: Status updated confirmation
    TrainingWorker->>JobQueue: Acknowledge job completion
```

### 5.4. Entity-Relationship (ER) Diagram
```mermaid
erDiagram
    USER ||--o{ DATASET : owns
    USER ||--o{ MODEL : owns
    USER ||--o{ TRAINING_JOB : initiates

    DATASET {
        VARCHAR id PK
        VARCHAR name
        TEXT description
        VARCHAR source_type
        TEXT source_path
        VARCHAR status
        TIMESTAMP created_at
        VARCHAR owner_id FK
    }

    MODEL {
        VARCHAR id PK
        VARCHAR name
        TEXT description
        VARCHAR model_type
        VARCHAR base_architecture
        TEXT source_path
        VARCHAR status
        TIMESTAMP created_at
        VARCHAR owner_id FK
    }

    TRAINING_JOB {
        VARCHAR id PK
        VARCHAR model_id FK
        VARCHAR dataset_id FK
        JSONB training_params
        VARCHAR status
        REAL progress
        VARCHAR owner_id FK
        TIMESTAMP started_at
        TIMESTAMP completed_at
        TEXT logs_path
        TEXT results_path
    }

    TRAINING_JOB ||--|{ MODEL : uses
    TRAINING_JOB ||--|{ DATASET : uses

    USER {
        VARCHAR id PK
        VARCHAR username
        VARCHAR email
        VARCHAR password_hash
        VARCHAR roles
    }
```

### 5.5. Use Case Diagram (Improved)
```mermaid
UMLet
Actor(User)

UseCase(Register Account)
UseCase(Login to System)
UseCase(Manage Datasets)
UseCase(Manage Models)
UseCase(Initiate Training)
UseCase(Monitor Training Progress)
UseCase(Stop Training)

User -- (Register Account)
User -- (Login to System)
User -- (Manage Datasets)
User -- (Manage Models)
User -- (Initiate Training)
User -- (Monitor Training Progress)
User -- (Stop Training)

(Manage Datasets) <. (Add Dataset) : <<include>>
(Manage Datasets) <. (View Dataset Details) : <<include>>
(Manage Datasets) <. (Delete Dataset) : <<include>>

(Manage Models) <. (Add Model) : <<include>>
(Manage Models) <. (View Model Details) : <<include>>
(Manage Models) <. (Delete Model) : <<include>>

(Initiate Training) <. (Select Dataset) : <<include>>
(Initiate Training) <. (Select Model) : <<include>>
(Initiate Training) <. (Configure Training Parameters) : <<include>>

```

## 6. Conclusion
This document provides a foundational software specification for the QLoRA Orchestration Service, detailing its requirements, architectural design, and key modules. By adhering to these specifications, the development team can build a robust, scalable, and maintainable system capable of efficiently managing and orchestrating QLoRA fine-tuning processes for video-text models.