# Frontend UI Specification: Orchestration Website

## 1. Introduction
This document outlines the technical specifications and architectural plan for the frontend user interface of the QLoRA orchestration website. The primary objective is to provide an intuitive and robust platform for users to manage datasets, configure and deploy QLoRA-enabled video models, and initiate training processes. Adhering to modern web development best practices, this specification details the chosen technology stack, core functionalities, and design principles.

## 2. Technology Stack
To ensure an industry-grade, scalable, and maintainable frontend, the following technologies have been selected:

-   **Framework:** [Next.js](https://nextjs.org/) (React Framework for Production)
    -   **Rationale:** Provides server-side rendering (SSR) and static site generation (SSG) for performance and SEO, API routes for backend integration, and a robust development experience.
-   **Styling:** [Tailwind CSS](https://tailwindcss.com/)
    -   **Rationale:** A utility-first CSS framework that enables rapid UI development, highly customizable designs, and ensures consistency across the application without writing custom CSS.
-   **Component Library:** [Shadcn/ui](https://ui.shadcn.com/)
    -   **Rationale:** A collection of re-usable components built with Radix UI and Tailwind CSS. It offers unstyled, accessible components that can be easily customized to fit the project's design system, promoting consistency and accelerating development.
-   **Animation/Effects Library:** [Aceternity UI](https://ui.aceternity.com/)
    -   **Rationale:** Provides visually appealing and modern UI components and animations that can enhance user experience and provide a polished, professional aesthetic.
-   **Language:** [TypeScript](https://www.typescriptlang.org/)
    -   **Rationale:** A superset of JavaScript that adds static typing, improving code quality, readability, and maintainability, especially in larger codebases.

## 3. Core Features and User Flow

### 3.1. User Authentication

#### 3.1.1. Login Page
-   **Purpose:** Allows existing users to access their dashboard.
-   **Components:** Input fields for `Email/Username` and `Password`, a `Login` button, and a link to the `Signup` page.
-   **Validation:** Client-side and server-side validation for input fields.
-   **Error Handling:** Display clear error messages for invalid credentials or network issues.

#### 3.1.2. Signup Page
-   **Purpose:** Enables new users to create an account.
-   **Components:** Input fields for `Username`, `Email`, `Password`, and `Confirm Password`, a `Signup` button, and a link to the `Login` page.
-   **Validation:** Password strength requirements, email format validation, and confirmation password matching.
-   **Error Handling:** Informative messages for existing users or invalid inputs.

### 3.2. Dashboard
Upon successful authentication, users are redirected to a personalized dashboard.

-   **Layout:** A clean, intuitive layout providing quick access to key functionalities.
-   **Navigation:** A persistent sidebar or top navigation bar with links to:
    -   `Dashboard Overview`
    -   `Dataset Management`
    -   `Model Management`
    -   `Training Jobs`
    -   `User Profile/Settings`
-   **Overview Section:** A summary of recent activities, active training jobs, and quick statistics (e.g., number of datasets, models).

### 3.3. Dataset Management

#### 3.3.1. Add New Dataset
-   **Purpose:** Facilitates the ingestion of new video datasets for training.
-   **Components:** A form or modal with fields for:
    -   `Dataset Name` (text input)
    -   `Description` (textarea)
    -   `Source Type` (dropdown: e.g., `Upload File(s)`, `URL`, `Cloud Storage Link`)
    -   Conditional input fields based on `Source Type` (e.g., file upload component, URL input).
-   **Actions:** `Submit` (to initiate upload/ingestion), `Cancel`.
-   **Progress Indicator:** For file uploads, display a progress bar.

#### 3.3.2. View/Manage Datasets
-   **Purpose:** Displays a list of all available datasets and allows for their management.
-   **Components:** A data table or card view showing:
    -   `Dataset Name`
    -   `Size`
    -   `Date Added`
    -   `Status` (e.g., `Ready`, `Processing`, `Error`)
-   **Actions per Dataset:** `View Details`, `Edit Metadata`, `Delete` (with confirmation modal).
-   **Search/Filter:** Functionality to search and filter datasets by name or status.

### 3.4. Model Management

#### 3.4.1. Add New Model
-   **Purpose:** Allows users to register new QLoRA-compatible video models.
-   **Components:** A form or modal with fields for:
    -   `Model Name` (text input)
    -   `Description` (textarea)
    -   `Model Type` (dropdown: e.g., `Video Classification`, `Video Captioning`, `Video Retrieval`)
    -   `Base Model Architecture` (dropdown: e.g., `Flamingo`, `Video-LLaMA`, `Custom`)
    -   `Model Source` (e.g., `Upload Model Weights`, `Hugging Face Hub ID`, `Custom Path`)
-   **Actions:** `Submit`, `Cancel`.

#### 3.4.2. View/Manage Models
-   **Purpose:** Displays a list of registered models and enables their management.
-   **Components:** A data table or card view showing:
    -   `Model Name`
    -   `Type`
    -   `Base Architecture`
    -   `Status` (e.g., `Ready`, `In Use`)
-   **Actions per Model:** `View Details`, `Edit Metadata`, `Delete` (with confirmation).
-   **Search/Filter:** Functionality to search and filter models.

### 3.5. Training Initiation and Monitoring

#### 3.5.1. Start New Training
-   **Purpose:** Guides users through the process of configuring and launching a new QLoRA fine-tuning job.
-   **Components:** A multi-step wizard or form:
    -   **Step 1: Select Dataset:** Dropdown/selection from available datasets.
    -   **Step 2: Select Model:** Dropdown/selection from available models.
    -   **Step 3: Configure Training Parameters:**
        -   `Training Type` (e.g., `Fine-tuning`, `Pre-training`)
        -   `QLoRA Parameters` (e.g., `Rank (r)`, `Alpha`, `Dropout`, `Target Modules`)
        -   `Optimization Parameters` (e.g., `Learning Rate`, `Batch Size`, `Epochs`)
        -   `Output Directory/Name`
    -   **Step 4: Review & Launch:** Summary of configurations before final submission.
-   **Actions:** `Next`, `Back`, `Launch Training`, `Cancel`.

#### 3.5.2. Training Job Monitoring
-   **Purpose:** Provides real-time updates and historical data for training jobs.
-   **Components:** A dashboard section or dedicated page showing:
    -   List of active and completed training jobs.
    -   `Job ID`, `Model`, `Dataset`, `Status` (`Running`, `Completed`, `Failed`), `Progress` (percentage/epochs).
    -   **Details View:** For an individual job, display training logs, loss curves, metric plots, and resource utilization (if available).
-   **Actions:** `Stop Training` (for active jobs), `View Logs`, `Download Results`.

## 4. Design Principles
-   **User-Centric:** Intuitive navigation, clear feedback, and minimal cognitive load.
-   **Responsive:** Optimized for various screen sizes (desktop, tablet, mobile).
-   **Accessible:** Adherence to WCAG guidelines for inclusivity.
-   **Modular:** Component-based architecture for reusability and maintainability.
-   **Performant:** Fast loading times and smooth interactions.
-   **Visually Appealing:** Modern, clean design utilizing Shadcn/ui and Aceternity UI for a polished look.

## 5. API Integration
The frontend will interact with a FastAPI backend for all data operations (authentication, dataset/model management, training job orchestration). All API calls will be asynchronous and handle appropriate error states.

## 6. Future Enhancements
-   Real-time progress graphs for training metrics.
-   Notification system for job completion or failures.
-   Role-based access control.
-   Integration with external data sources (e.g., S3, Google Cloud Storage).
-   Advanced model versioning and deployment features.