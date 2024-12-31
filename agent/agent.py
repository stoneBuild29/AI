from huggingface_integration import generate_text, load_huggingface_model, classify_text, load_classifier_model
from task_manager import TaskManager
from transformers import DistilBertForSequenceClassification
import torch
#from transformers import logging
from transformers import pipeline
#logging.set_verbosity_error()

classifier = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    ignore_mismatched_sizes=True
)


class AIAgentWithExtendedFeatures:
    def __init__(self, name):
        self.name = name
        self.memory = {}
        self.history = []
        self.task_manager = TaskManager()
        
        # Load Hugging Face models
        self.model, self.tokenizer = load_huggingface_model("gpt2")
        # self.classifier = load_classifier_model("distilbert-base-uncased")

        # Initialize a classification pipeline with device set
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
        )

    def profile(self, task):
        """ Profile the task and set the environment """
        print(f"{self.name} is profiling task: {task}")
        self.memory['task'] = task

    def plan(self, goal):
        """ Use Hugging Face model to help plan or generate decisions """
        print(f"{self.name} is planning to achieve goal: {goal}")
        
        # Use Hugging Face model to generate text (this could represent a decision-making process)
        prompt = f"Plan for achieving the goal: {goal}"
        action = generate_text(self.model, self.tokenizer, prompt)
        self.history.append(f"Planned: {action}")
        return action


    def classify_and_prioritize(self, task):
        """ Classify the task and prioritize based on the model """
        print(f"{self.name} is classifying and prioritizing task: {task}")
        
        # Classify task type and set priority using the pipeline
        classification = self.classifier(task)
        priority = 5 if classification[0]['label'] == 'POSITIVE' else 1
        self.task_manager.add_task(task, priority)

        # Get the highest priority task
        highest_priority_task = self.task_manager.get_highest_priority_task()
        return highest_priority_task


    def execute(self, action):
        """ Execute the planned action """
        print(f"{self.name} is executing action: {action}")
        self.history.append(f"Executed: {action}")
        return f"{action} completed successfully"

    def learn_from_feedback(self):
        """ Feedback loop to improve decision-making """
        feedback = "positive"  # Simulated feedback
        print(f"{self.name} received feedback: {feedback}")

    def run(self, task, goal):
        """ Run the agent cycle: Profiling, Planning, Executing, and Feedback """
        self.profile(task)
        action = self.plan(goal)
        prioritized_task = self.classify_and_prioritize(task)
        result = self.execute(action)
        print(f"Prioritized Task: {prioritized_task}")
        self.learn_from_feedback()
    




# Example usage: Running the agent
if __name__ == "__main__":
    agent = AIAgentWithExtendedFeatures(name="AI_Agent_With_Extended_Features")
    agent.run("Build a Secure Blockchain System", "Deploy Smart Contracts")


