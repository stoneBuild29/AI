class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, task, priority=1):
        """ Add a new task with priority (1-5) """
        self.tasks.append({"task": task, "priority": priority})
        print(f"Task '{task}' added with priority {priority}.")

    def get_highest_priority_task(self):
        """ Get the task with the highest priority """
        if self.tasks:
            highest_priority_task = max(self.tasks, key=lambda x: x["priority"])
            return highest_priority_task["task"]
        else:
            return None

    def list_tasks(self):
        """ List all tasks with their priorities """
        for task in self.tasks:
            print(f"Task: {task['task']} - Priority: {task['priority']}")