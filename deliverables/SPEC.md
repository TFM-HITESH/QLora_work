# API Specification: QLoRA Orchestration Service

## 1. Introduction
This document provides a basic API specification for the QLoRA Orchestration Service backend, implemented using FastAPI. The API enables external clients, including the frontend UI, to interact with the service for user authentication, managing datasets and models, and orchestrating QLoRA fine-tuning jobs. This specification is designed to be easily convertible into Mintlify documentation.

## 2. Base URL
`https://api.qlora-orchestration.com/v1` (Conceptual)

## 3. Authentication
All protected endpoints require a Bearer Token (JWT) obtained after successful login.

### 3.1. User Login
-   **Endpoint:** `/auth/login`
-   **Method:** `POST`
-   **Description:** Authenticates a user and returns an access token.
-   **Request Body:**
    ```json
    {
      "username": "string",
      "password": "string"
    }
    ```
-   **Responses:**
    -   `200 OK`:
        ```json
        {
          "access_token": "string",
          "token_type": "bearer"
        }
        ```
    -   `401 Unauthorized`: Invalid credentials.

### 3.2. User Signup
-   **Endpoint:** `/auth/signup`
-   **Method:** `POST`
-   **Description:** Registers a new user account.
-   **Request Body:**
    ```json
    {
      "username": "string",
      "email": "string",
      "password": "string"
    }
    ```
-   **Responses:**
    -   `201 Created`:
        ```json
        {
          "message": "User registered successfully"
        }
        ```
    -   `400 Bad Request`: User already exists or invalid input.

## 4. Dataset Management

### 4.1. Initiate Dataset Upload
-   **Endpoint:** `/datasets/upload-initiate`
-   **Method:** `POST`
-   **Description:** Initiates a dataset upload process, returning a pre-signed S3 URL for direct client-to-S3 upload.
-   **Authentication:** Required
-   **Request Body:**
    ```json
    {
      "file_name": "string",
      "file_type": "string",
      "dataset_name": "string",
      "dataset_description": "string"
    }
    ```
-   **Responses:**
    -   `200 OK`:
        ```json
        {
          "upload_url": "string",
          "dataset_id": "string",
          "s3_key": "string"
        }
        ```
    -   `400 Bad Request`: Invalid input.

### 4.2. Complete Dataset Upload
-   **Endpoint:** `/datasets/upload-complete`
-   **Method:** `POST`
-   **Description:** Notifies the backend that the S3 upload is complete, allowing the system to process the dataset.
-   **Authentication:** Required
-   **Request Body:**
    ```json
    {
      "dataset_id": "string",
      "s3_key": "string"
    }
    ```
-   **Responses:**
    -   `200 OK`:
        ```json
        {
          "message": "Dataset processing initiated",
          "dataset_id": "string"
        }
        ```
    -   `400 Bad Request`: Invalid input or S3 key mismatch.

### 4.3. Get All Datasets
-   **Endpoint:** `/datasets`
-   **Method:** `GET`
-   **Description:** Retrieves a list of all datasets accessible by the user.
-   **Authentication:** Required
-   **Responses:**
    -   `200 OK`:
        ```json
        [
          {
            "id": "string",
            "name": "string",
            "description": "string",
            "s3_path": "string",
            "status": "ready" | "processing" | "error",
            "created_at": "datetime"
          }
        ]
        ```

### 4.4. Get Dataset by ID
-   **Endpoint:** `/datasets/{dataset_id}`
-   **Method:** `GET`
-   **Description:** Retrieves details of a specific dataset.
-   **Authentication:** Required
-   **Responses:**
    -   `200 OK`:
        ```json
        {
          "id": "string",
          "name": "string",
          "description": "string",
          "s3_path": "string",
          "status": "ready" | "processing" | "error",
          "created_at": "datetime",
          "file_count": "integer",
          "total_size_bytes": "integer"
        }
        ```
    -   `404 Not Found`: Dataset not found.

### 4.5. Delete Dataset
-   **Endpoint:** `/datasets/{dataset_id}`
-   **Method:** `DELETE`
-   **Description:** Deletes a dataset and its associated files from S3.
-   **Authentication:** Required
-   **Responses:**
    -   `204 No Content`: Dataset deleted successfully.
    -   `404 Not Found`: Dataset not found.

## 5. Model Management

### 5.1. Get Allowed Models
-   **Endpoint:** `/models/allowed`
-   **Method:** `GET`
-   **Description:** Retrieves a list of pre-defined and allowed QLoRA-compatible models available for fine-tuning.
-   **Authentication:** Required
-   **Responses:**
    -   `200 OK`:
        ```json
        [
          {
            "id": "string",
            "name": "string",
            "description": "string",
            "model_type": "video_classification" | "video_captioning",
            "base_architecture": "flamingo" | "video_llama",
            "version": "string"
          }
        ]
        ```

### 5.2. Get Model by ID
-   **Endpoint:** `/models/{model_id}`
-   **Method:** `GET`
-   **Description:** Retrieves details of a specific allowed model.
-   **Authentication:** Required
-   **Responses:**
    -   `200 OK`:
        ```json
        {
          "id": "string",
          "name": "string",
          "description": "string",
          "model_type": "video_classification" | "video_captioning",
          "base_architecture": "flamingo" | "video_llama",
          "version": "string",
          "supported_lora_params": {
            "min_rank": "integer",
            "max_rank": "integer",
            "default_alpha": "integer"
          }
        }
        ```
    -   `404 Not Found`: Model not found.

## 6. Training Orchestration

### 6.1. Start New Training Job
-   **Endpoint:** `/trainings`
-   **Method:** `POST`
-   **Description:** Initiates a new QLoRA fine-tuning job.
-   **Authentication:** Required
-   **Request Body:**
    ```json
    {
      "model_id": "string",
      "dataset_id": "string",
      "training_params": {
        "epochs": "integer",
        "learning_rate": "float",
        "batch_size": "integer",
        "lora_rank": "integer",
        "lora_alpha": "integer",
        "lora_dropout": "float",
        "target_modules": "array of strings" // e.g., ["q_proj", "v_proj"]
      },
      "output_name": "string"
    }
    ```
-   **Responses:**
    -   `202 Accepted`:
        ```json
        {
          "job_id": "string",
          "status": "queued",
          "message": "Training job initiated successfully"
        }
        ```
    -   `400 Bad Request`: Invalid parameters or model/dataset not found.

### 6.2. Get All Training Jobs
-   **Endpoint:** `/trainings`
-   **Method:** `GET`
-   **Description:** Retrieves a list of all training jobs for the user.
-   **Authentication:** Required
-   **Responses:**
    -   `200 OK`:
        ```json
        [
          {
            "job_id": "string",
            "model_id": "string",
            "dataset_id": "string",
            "status": "queued" | "running" | "completed" | "failed",
            "progress": "float", // 0.0 to 1.0
            "started_at": "datetime",
            "completed_at": "datetime" | null,
            "output_path": "string" | null
          }
        ]
        ```

### 6.3. Get Training Job Status
-   **Endpoint:** `/trainings/{job_id}`
-   **Method:** `GET`
-   **Description:** Retrieves the status and details of a specific training job.
-   **Authentication:** Required
-   **Responses:**
    -   `200 OK`:
        ```json
        {
          "job_id": "string",
          "model_id": "string",
          "dataset_id": "string",
          "status": "queued" | "running" | "completed" | "failed",
          "progress": "float",
          "started_at": "datetime",
          "completed_at": "datetime" | null,
          "output_path": "string" | null,
          "logs": "string" // Recent logs or a link to full logs
        }
        ```
    -   `404 Not Found`: Job not found.

### 6.4. Stop Training Job
-   **Endpoint:** `/trainings/{job_id}/stop`
-   **Method:** `POST`
-   **Description:** Requests to stop a running training job.
-   **Authentication:** Required
-   **Responses:**
    -   `202 Accepted`:
        ```json
        {
          "message": "Stop request received for job_id"
        }
        ```
    -   `404 Not Found`: Job not found.
    -   `400 Bad Request`: Job is not in a stoppable state.

## 7. Data Specifications

### 7.1. Dataset Storage Specification (S3)
User-uploaded datasets will be stored in a dedicated S3 bucket. Each dataset will reside in a unique folder identified by its `dataset_id`.

-   **Bucket Structure:** `s3://[your-bucket-name]/datasets/{dataset_id}/`
-   **File Naming:** Original file names will be preserved within the dataset folder.
-   **Metadata:** Dataset metadata (name, description, status, S3 path) will be stored in the PostgreSQL database, linked by `dataset_id`.

**Example S3 Path:** `s3://qlora-orchestration-datasets/datasets/a1b2c3d4-e5f6-7890-1234-567890abcdef/my_video_dataset.zip`

### 7.2. Model Storage Specification (S3)
Pre-defined models will also be stored in S3, typically in a read-only bucket accessible by the training workers.

-   **Bucket Structure:** `s3://[your-bucket-name]/models/{model_id}/`
-   **Contents:** Model weights, configuration files, and any other necessary assets for loading and running the model.

**Example S3 Path:** `s3://qlora-orchestration-models/models/flamingo-v1/`

## 8. Error Handling
Standard HTTP status codes will be used. Error responses will typically include a JSON body with an `"detail"` field providing a descriptive error message.

```json
{
  "detail": "Error description here."
}
```