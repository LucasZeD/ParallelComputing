# Data-set author:

*[Original Data-set - Atif Masih](https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades)

# Environment

1. move to folder
```bash
cd /Projeto2/data-set
```
2. create
```bash
python3 -m venv env_name
```
3. activate environment
- WIN
```bash
env\Scripts\activate
```
- UNIX
```bash
source env_name/bin/activate
```
4. deactivate environment
```bash
deactivate
```

# Run 

1. install depedencies
```bash
pip3 install -r requirements.txt
```
2. use .ipynb or .py to preprocess data
Files with preprocessed data will be saved at Projeto1/PrepData
```bash
python3 preprocess.py
```

---

# Files Structure

- `preprocess.ipynb`    -> python notebook with preprocessing code of data and instructions (only for educational purpose, do not compile from it)
- `StudentData.csv`     -> original data
- `Projeto1/PrepData`   -> store preprocessed input data
- `_oversampled_train/test.txt` -> store oversampled data needed to achieve 10 seconds of sequential code run-time
- `.txt files`          -> files read by the c++ id3 codes

# Dataset Description:

- `age`: Numerical value representing the student's age.
- `gender`: Categorical attribute indicating the student's gender (M for Male, F for Female).
- `parental_education`: The highest level of education achieved by the student's parents, such as High School or Some College.
- `family_income`: Numerical value representing the household income level.
- `previous_grades`: The student's past academic performance categorized as A, B, or C.
attendance: Numerical percentage of class attendance by the student.
- `class_participation`: Level of engagement in class activities, classified as Low, Medium, or High.
- `study_hours`: Average number of study hours per week.
- `major`: Field of study chosen by the student.
- `uni_type`: Type of university attended, either Public or Private.
- `financial_status`: The economic condition of the student, categorized as Low, Medium, or High.
- `parental_involvement`: The degree of involvement of parents in the student's education, categorized as Low, Medium, or High.
- `educational_resources`: Availability of learning materials and resources at home, categorized as Yes or No.
- `motivation`: The level of motivation for academics, classified as Low, Medium, or High.
- `self_esteem`: The level of confidence, categorized as Low, Medium, or High.
- `stress_levels`: The amount of stress experienced by the student, categorized as Low, Medium, or High.
- `school_environment`: The student's perception of their school environment as Negative, Neutral, or Positive.
- `professor_quality`: Assessment of professors' teaching quality, classified as Low, Medium, or High.
- `class_size`: Numerical value indicating the number of students in a class.
- `extracurricular_activities`: Participation in non-academic activities, categorized as Yes or No.
- `sleep_patterns`: Average hours of sleep per day.
- `nutrition`: Quality of diet, classified as Unhealthy, Balanced, or Healthy.
- `physical_activity`: Level of physical activity, categorized as Low, Medium, or High.
- `screen_time`: Number of hours spent on screen-based activities daily.
- `educational_tech_use`: Use of educational technologies, classified as Yes or No.
- `peer_group`: Influence of peers, categorized as Negative, Neutral, or Positive.
- `bullying`: Whether the student experienced bullying, categorized as Yes or No.
- `study_space`: Availability of a dedicated study area at home, categorized as Yes or No.
- `learning_style`: Preferred way of learning, such as Visual, Auditory, or Kinesthetic.
- `tutoring`: Participation in tutoring programs, categorized as Yes or No.
- `mentoring`: Access to mentoring support, categorized as Yes or No.
- `lack_of_interest`: Level of interest in academics, categorized as Low, Medium, or High.
- `time_wasted_on_social_media`: Daily time spent on social media platforms.
- `sports_participation`: Involvement in sports activities, categorized as Low, Medium, or High.
- `grades`: Final academic grades, categorized as A, B, or C.