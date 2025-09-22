"""
Quiz Service - Handles MCQ quiz generation and evaluation for ProfAI
"""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional
from services.llm_service import LLMService
from models.schemas import Quiz, QuizQuestion, QuizSubmission, QuizResult, QuizDisplay, QuizQuestionDisplay
import config

class QuizService:
    """Service for generating and evaluating MCQ quizzes."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.quiz_storage_dir = os.path.join(os.path.dirname(__file__), "..", "data", "quizzes")
        self.answers_storage_dir = os.path.join(os.path.dirname(__file__), "..", "data", "quiz_answers")
        
        # Ensure storage directories exist
        os.makedirs(self.quiz_storage_dir, exist_ok=True)
        os.makedirs(self.answers_storage_dir, exist_ok=True)
        
        logging.info("QuizService initialized")
    
    async def generate_module_quiz(self, module_week: int, course_content: dict) -> Quiz:
        """Generate a 20-question MCQ quiz for a specific module."""
        try:
            # Find the specific module
            module = None
            for mod in course_content.get("modules", []):
                if mod.get("week") == module_week:
                    module = mod
                    break
            
            if not module:
                raise ValueError(f"Module week {module_week} not found in course content")
            
            # Extract content for the module
            module_content = self._extract_module_content(module)
            
            # Generate quiz using LLM
            quiz_id = f"module_{module_week}_{uuid.uuid4().hex[:8]}"
            
            logging.info(f"Generating 20-question quiz for module week {module_week}")
            
            quiz_prompt = self._create_module_quiz_prompt(module, module_content)
            quiz_response = await self.llm_service.generate_response(quiz_prompt, temperature=0.7)
            
            # Parse the LLM response into structured quiz
            questions = self._parse_quiz_response(quiz_response, quiz_id)
            
            # Ensure we have exactly 20 questions
            if len(questions) < 20:
                # Generate additional questions if needed
                additional_prompt = self._create_additional_questions_prompt(module_content, 20 - len(questions))
                additional_response = await self.llm_service.generate_response(additional_prompt, temperature=0.7)
                additional_questions = self._parse_quiz_response(additional_response, quiz_id, start_id=len(questions))
                questions.extend(additional_questions)
            
            # Take only first 20 questions
            questions = questions[:20]
            
            quiz = Quiz(
                quiz_id=quiz_id,
                title=f"Module {module_week} Quiz: {module.get('title', 'Module Quiz')}",
                description=f"20-question MCQ quiz covering content from Week {module_week}",
                questions=questions,
                total_questions=len(questions),
                quiz_type="module",
                module_week=module_week
            )
            
            # Store quiz and answers with course_id
            self._store_quiz(quiz, course_content.get('course_id'))
            
            logging.info(f"Generated module quiz with {len(questions)} questions")
            return quiz
            
        except Exception as e:
            logging.error(f"Error generating module quiz: {e}")
            raise e
    
    async def generate_course_quiz(self, course_content: dict) -> Quiz:
        """Generate a 40-question MCQ quiz covering the entire course."""
        try:
            # Extract content from all modules
            all_content = self._extract_all_course_content(course_content)
            
            quiz_id = f"course_{uuid.uuid4().hex[:8]}"
            
            logging.info("Generating 40-question course quiz")
            
            # Generate quiz using LLM in chunks (20 questions at a time for better results)
            quiz_prompt_1 = self._create_course_quiz_prompt(all_content, part=1)
            quiz_response_1 = await self.llm_service.generate_response(quiz_prompt_1, temperature=0.7)
            questions_1 = self._parse_quiz_response(quiz_response_1, quiz_id, start_id=0)
            
            quiz_prompt_2 = self._create_course_quiz_prompt(all_content, part=2)
            quiz_response_2 = await self.llm_service.generate_response(quiz_prompt_2, temperature=0.7)
            questions_2 = self._parse_quiz_response(quiz_response_2, quiz_id, start_id=20)
            
            # Combine all questions
            all_questions = questions_1 + questions_2
            
            # Take exactly 40 questions
            all_questions = all_questions[:40]
            
            quiz = Quiz(
                quiz_id=quiz_id,
                title=f"Final Course Quiz: {course_content.get('course_title', 'Course Quiz')}",
                description="40-question comprehensive MCQ quiz covering the entire course content",
                questions=all_questions,
                total_questions=len(all_questions),
                quiz_type="course",
                module_week=None
            )
            
            # Store quiz and answers with course_id
            self._store_quiz(quiz, course_content.get('course_id'))
            
            logging.info(f"Generated course quiz with {len(all_questions)} questions")
            return quiz
            
        except Exception as e:
            logging.error(f"Error generating course quiz: {e}")
            raise e
    
    def evaluate_quiz(self, submission: QuizSubmission) -> QuizResult:
        """Evaluate a quiz submission and return results."""
        try:
            # Load quiz and answers
            quiz_data = self._load_quiz_answers(submission.quiz_id)
            if not quiz_data:
                raise ValueError(f"Quiz {submission.quiz_id} not found")
            
            correct_answers = quiz_data["answers"]
            quiz_info = quiz_data["quiz"]
            
            # Calculate score
            score = 0
            detailed_results = []
            
            for question_id, user_answer in submission.answers.items():
                is_correct = user_answer.upper() == correct_answers.get(question_id, "").upper()
                if is_correct:
                    score += 1
                
                detailed_results.append({
                    "question_id": question_id,
                    "user_answer": user_answer.upper(),
                    "correct_answer": correct_answers.get(question_id, ""),
                    "is_correct": is_correct
                })
            
            total_questions = len(correct_answers)
            percentage = (score / total_questions) * 100 if total_questions > 0 else 0
            passed = percentage >= 60.0
            
            result = QuizResult(
                quiz_id=submission.quiz_id,
                user_id=submission.user_id,
                score=score,
                total_questions=total_questions,
                percentage=round(percentage, 2),
                passed=passed,
                detailed_results=detailed_results
            )
            
            # Store submission result
            self._store_submission_result(submission, result)
            
            logging.info(f"Evaluated quiz {submission.quiz_id}: {score}/{total_questions} ({percentage:.1f}%)")
            return result
            
        except Exception as e:
            logging.error(f"Error evaluating quiz: {e}")
            raise e
    
    def get_quiz_without_answers(self, quiz_id: str) -> Optional[QuizDisplay]:
        """Get quiz for display (without correct answers)."""
        try:
            quiz_file = os.path.join(self.quiz_storage_dir, f"{quiz_id}.json")
            if not os.path.exists(quiz_file):
                return None
            
            with open(quiz_file, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
            
            # Create display questions without correct answers
            display_questions = []
            for question in quiz_data["questions"]:
                display_question = QuizQuestionDisplay(
                    question_id=question["question_id"],
                    question_text=question["question_text"],
                    options=question["options"],
                    topic=question.get("topic", "")
                )
                display_questions.append(display_question)
            
            # Create display quiz
            display_quiz = QuizDisplay(
                quiz_id=quiz_data["quiz_id"],
                title=quiz_data["title"],
                description=quiz_data["description"],
                questions=display_questions,
                total_questions=quiz_data["total_questions"],
                quiz_type=quiz_data["quiz_type"],
                module_week=quiz_data.get("module_week"),
                course_id=quiz_data.get("course_id")
            )
            
            return display_quiz
            
        except Exception as e:
            logging.error(f"Error loading quiz {quiz_id}: {e}")
            return None
    
    def _extract_module_content(self, module: dict) -> str:
        """Extract text content from a module."""
        content_parts = [f"Module Week {module.get('week')}: {module.get('title', '')}"]
        
        for sub_topic in module.get("sub_topics", []):
            content_parts.append(f"\n--- {sub_topic.get('title', '')} ---")
            content_parts.append(sub_topic.get('content', ''))
        
        return "\n".join(content_parts)
    
    def _extract_all_course_content(self, course_content: dict) -> str:
        """Extract text content from entire course."""
        content_parts = [f"Course: {course_content.get('course_title', '')}"]
        
        for module in course_content.get("modules", []):
            content_parts.append(self._extract_module_content(module))
        
        return "\n".join(content_parts)
    
    def _create_module_quiz_prompt(self, module: dict, content: str) -> str:
        """Create prompt for module quiz generation."""
        return f"""Generate a 20-question multiple choice quiz based on the following module content.

MODULE INFORMATION:
Week: {module.get('week')}
Title: {module.get('title', '')}

CONTENT:
{content}

REQUIREMENTS:
1. Generate exactly 20 multiple choice questions
2. Each question should have 4 options (A, B, C, D)
3. Questions should cover different aspects of the module content
4. Mix difficulty levels: 40% easy, 40% medium, 20% hard
5. Include practical application questions
6. Ensure questions test understanding, not just memorization

FORMAT YOUR RESPONSE AS:
Q1. [Question text]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
ANSWER: [A/B/C/D]
EXPLANATION: [Brief explanation]

Q2. [Next question...]

Continue this format for all 20 questions."""
    
    def _create_course_quiz_prompt(self, content: str, part: int) -> str:
        """Create prompt for course quiz generation."""
        question_range = f"questions {1 + (part-1)*20} to {part*20}" if part == 1 else f"questions 21 to 40"
        
        return f"""Generate {question_range} of a comprehensive multiple choice quiz based on the entire course content below.

COURSE CONTENT:
{content[:8000]}  # Limit content to avoid token limits

REQUIREMENTS:
1. Generate exactly 20 multiple choice questions for this part
2. Each question should have 4 options (A, B, C, D)
3. Cover content from all modules proportionally
4. Mix difficulty levels: 30% easy, 50% medium, 20% hard
5. Include synthesis questions that connect concepts across modules
6. Test both theoretical understanding and practical application

FORMAT YOUR RESPONSE AS:
Q{1 + (part-1)*20}. [Question text]
A) [Option A]
B) [Option B]
C) [Option C] 
D) [Option D]
ANSWER: [A/B/C/D]
EXPLANATION: [Brief explanation]

Continue this format for all 20 questions in this part."""
    
    def _create_additional_questions_prompt(self, content: str, num_questions: int) -> str:
        """Create prompt for generating additional questions."""
        return f"""Generate {num_questions} additional multiple choice questions based on the following content:

{content}

REQUIREMENTS:
1. Generate exactly {num_questions} questions
2. Each question should have 4 options (A, B, C, D)
3. Focus on different aspects not covered in previous questions
4. Maintain good difficulty distribution

FORMAT YOUR RESPONSE AS:
Q. [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
ANSWER: [A/B/C/D]
EXPLANATION: [Brief explanation]"""
    
    def _parse_quiz_response(self, response: str, quiz_id: str, start_id: int = 0) -> List[QuizQuestion]:
        """Parse LLM response into structured quiz questions."""
        questions = []
        lines = response.split('\n')
        
        current_question = {}
        question_count = start_id
        
        for line in lines:
            line = line.strip()
            
            # Question line
            if line.startswith('Q') and ('.' in line or ')' in line):
                if current_question and 'question_text' in current_question:
                    # Save previous question
                    questions.append(self._create_question_object(current_question, quiz_id, question_count))
                    question_count += 1
                
                # Start new question
                current_question = {}
                # Extract question text after Q1., Q2., etc.
                question_text = line.split('.', 1)[-1].strip() if '.' in line else line.split(')', 1)[-1].strip()
                current_question['question_text'] = question_text
                current_question['options'] = []
            
            # Option lines
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                if 'options' in current_question:
                    option_text = line[2:].strip()  # Remove "A)" prefix
                    current_question['options'].append(option_text)
            
            # Answer line
            elif line.startswith('ANSWER:'):
                answer = line.replace('ANSWER:', '').strip().upper()
                current_question['correct_answer'] = answer
            
            # Explanation line
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
                current_question['explanation'] = explanation
        
        # Don't forget the last question
        if current_question and 'question_text' in current_question:
            questions.append(self._create_question_object(current_question, quiz_id, question_count))
        
        return questions
    
    def _create_question_object(self, question_data: dict, quiz_id: str, question_num: int) -> QuizQuestion:
        """Create a QuizQuestion object from parsed data."""
        return QuizQuestion(
            question_id=f"{quiz_id}_q{question_num + 1}",
            question_text=question_data.get('question_text', ''),
            options=question_data.get('options', [])[:4],  # Ensure max 4 options
            correct_answer=question_data.get('correct_answer', 'A'),
            explanation=question_data.get('explanation', ''),
            topic=question_data.get('topic', '')
        )
    
    def _store_quiz(self, quiz: Quiz, course_id: str = None):
        """Store quiz and answers separately."""
        try:
            # Store full quiz data with course_id
            quiz_data = quiz.dict()
            if course_id:
                quiz_data['course_id'] = str(course_id)
            
            quiz_file = os.path.join(self.quiz_storage_dir, f"{quiz.quiz_id}.json")
            with open(quiz_file, 'w', encoding='utf-8') as f:
                json.dump(quiz_data, f, indent=2, ensure_ascii=False)
            
            # Store answers separately for evaluation
            answers = {}
            for question in quiz.questions:
                answers[question.question_id] = question.correct_answer
            
            answer_data = {
                "quiz_id": quiz.quiz_id,
                "answers": answers,
                "quiz": {
                    "title": quiz.title,
                    "total_questions": quiz.total_questions,
                    "quiz_type": quiz.quiz_type,
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            answer_file = os.path.join(self.answers_storage_dir, f"{quiz.quiz_id}_answers.json")
            with open(answer_file, 'w', encoding='utf-8') as f:
                json.dump(answer_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Stored quiz {quiz.quiz_id} and answers")
            
        except Exception as e:
            logging.error(f"Error storing quiz: {e}")
            raise e
    
    def _load_quiz_answers(self, quiz_id: str) -> Optional[dict]:
        """Load quiz answers for evaluation."""
        try:
            answer_file = os.path.join(self.answers_storage_dir, f"{quiz_id}_answers.json")
            if not os.path.exists(answer_file):
                return None
            
            with open(answer_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"Error loading quiz answers: {e}")
            return None
    
    def _store_submission_result(self, submission: QuizSubmission, result: QuizResult):
        """Store quiz submission and result."""
        try:
            submission_data = {
                "submission": submission.dict(),
                "result": result.dict(),
                "submitted_at": datetime.utcnow().isoformat()
            }
            
            submission_file = os.path.join(
                self.answers_storage_dir, 
                f"{submission.quiz_id}_{submission.user_id}_submission.json"
            )
            
            with open(submission_file, 'w', encoding='utf-8') as f:
                json.dump(submission_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Stored submission result for user {submission.user_id}")
            
        except Exception as e:
            logging.error(f"Error storing submission result: {e}")
