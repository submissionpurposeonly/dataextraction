import json
import time

class ShieldaInterceptor:
    def __init__(self, classifier, handler, escalator, logger):
        self.classifier = classifier
        self.handler = handler
        self.escalator = escalator
        self.logger = logger
        self.active_error = None  
        self.retry_count = 0      

    def process_step(self, step, task_name, thought, action, raw_obs, history, ground_truth):

        diagnosis = self.classifier.diagnose(thought, action, raw_obs, ground_truth)
        
        if not diagnosis or diagnosis.get("id") == "None":
            # 如果之前有错，现在好了，说明 Local Repair 成功了！
            if self.active_error:
                print(f"  [SHIELDA] ✅ Agent RECOVERED from {self.active_error['id']}!")

            self.logger.log_whitebox(step, task_name, thought, str(action), str(raw_obs), str(raw_obs), {"id": "None"}, {}, {})
            self.active_error = None
            self.retry_count = 0
            return raw_obs 

        print(f"  [SHIELDA] ⚠️ Active Error: {diagnosis['id']}")
        
        self.active_error = diagnosis 
        
        if self.retry_count == 0:
            self.retry_count = 1
        else:
            self.retry_count += 1

        if self.retry_count <= 3:
            print(f"  [Handler] 🛡️ Correction Attempt {self.retry_count}/3")
            
            recovery = self.handler.get_recovery_instruction(
                self.active_error['id'], 
                self.active_error.get('thinking', ''), 
                raw_obs, 
                self.retry_count
            )
            
            refined_obs = f"*** SYSTEM ALERT ***\n{recovery.get('prompt')}"
            self.logger.log_whitebox(step, task_name, thought, str(action), str(raw_obs), refined_obs, 
                                   self.active_error, {"status": "Activated", "output": recovery.get('prompt')}, {})
            return refined_obs

        else:
            print(f"  [Escalator] 💣 ROOT CAUSE INTERVENTION")
            
            deep_rca = self.escalator.perform_deep_analysis(history, self.classifier.taxonomy_str, ground_truth)
            
            strategic_obs = (f"*** CRITICAL OVERRIDE ***\n"
                             f"ROOT CAUSE: {deep_rca.get('exception_name')}\n"
                             f"INSTRUCTION: {deep_rca.get('strategic_prompt')}")
            
            self.logger.log_whitebox(step, task_name, thought, str(action), str(raw_obs), strategic_obs, 
                                   self.active_error, {"status": "Handover"}, 
                                   {"status": "Activated", "output": strategic_obs, "exception_name": deep_rca.get('exception_name')})

            self.active_error = None
            self.retry_count = 0
            return strategic_obs