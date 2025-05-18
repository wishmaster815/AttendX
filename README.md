# AttendX - AI-Powered Attendance System

An intelligent attendance system that uses face recognition to automatically mark attendance.

## Setup Instructions

1. Clone the repository:
```bash
git clone git@github.com:wishmaster815/AttendX.git
cd AttendX
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required model:
   - Create a `models` directory in the project root if it doesn't exist
   - Download the buffalo_l model from [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/python-package)
   - Extract the downloaded model into the `models` directory
   - The final structure should be: `models/buffalo_l/`

5. Run the application:
```bash
python gui_app.py
```

## Project Structure
- `faces/`: Contains face images for recognition
- `models/`: Should contain the buffalo_l model (not included in repo due to size)
- `attendance sheets/`: Generated attendance records
- `gui_app.py`: Main application file
- `requirements.txt`: Python dependencies

## Note
The model files are not included in the repository due to their large size. You need to download them separately as described in the setup instructions. 