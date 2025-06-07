import os
import re
import argparse
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Test: python generate_sample.py --model deepseek-ai/deepseek-coder-6.7b-instruct --budgets 1 --build_dir build_generation
args = argparse.ArgumentParser()
args.add_argument("--model", type=str, required=True)
args.add_argument("--budgets", type=int, required=True)
args.add_argument("--build_dir", type=str, required=True)
args.add_argument("--few_shot", type=int, default=0, choices=[0, 1, 2, 3, 4])


args = args.parse_args()


model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)


class VerilogTaskDataset(Dataset):
    def __init__(self, task_list):
        self.task_list = task_list

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        task = self.task_list[idx]
        return {
            "dataset": task["dataset"],
            "problem_name": task["problem_name"],
            "prompt": task["prompt"],
            "ref": task["ref"],
        }

    def _apply_prompt(self, prompt, few_shot: int = 0):
        if few_shot == 0:
            system_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates Verilog-only code based on the provided specifications.",
                        # "The generated code **must be syntactically correct and fully compatible with Icarus Verilog (iverilog)**. Do not include SystemVerilog features not supported by iverilog. Ensure all modules are self-contained and compilable.",
                        "Assume all sequential logic is triggered on the positive edge of the clock. The reset input is active high synchronous and should reset the finite-state machine to an appropriate initial state.",
                    ),
                }
            ]
        else:
            system_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates Verilog-only code based on the provided specifications.",
                        # "The generated code **must be syntactically correct and fully compatible with Icarus Verilog (iverilog)**. Do not include SystemVerilog features not supported by iverilog. Ensure all modules are self-contained and compilable.",
                        "Assume all sequential logic is triggered on the positive edge of the clock. The reset input is active high synchronous and should reset the finite-state machine to an appropriate initial state.",
                    ),
                }
            ]

        if few_shot == 1:
            system_prompt += self._few_shot_1()
        if few_shot == 2:
            system_prompt += self._few_shot_2()
        if few_shot == 3:
            system_prompt += self._few_shot_3()
        if few_shot == 4:
            system_prompt += self._few_shot_4()

        user_prompt = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return system_prompt + user_prompt

    def _few_shot_1(self):
        return [
            {
                "role": "uesr",
                "content": """Question:
Implement a hardware module named TopModule with the following interface.
All input and output ports are one bit unless otherwise specified.

 - input  in_ (8 bits)
 - output out (8 bits)

The module should implement an incrementer which increments the input by
one and writes the result to the output. Assume all values are encoded as
two's complement binary numbers.

Enclose your code with ```verilog and ```. Only output the code snippet
and do NOT output anything else.

Answer:
""",
            },
            {
                "role": "assistant",
                "content": """```verilog
module TopModule
(
  input  logic [7:0] in_,
  output logic [7:0] out
);

  // Combinational logic

  assign out = in_ + 1;

endmodule
```
""",
            },
        ]

    def _few_shot_2(self):
        return [
            {
                "role": "uesr",
                "content": """Question:
Implement a hardware module named TopModule with the following interface.
All input and output ports are one bit unless otherwise specified.

 - input  clk
 - input  reset
 - input  in_ (8 bits)
 - output out (8 bits)

The module should implement an 8-bit registered incrementer with an
active-high synchronous reset. The 8-bit input is first registered and
then incremented by one on the next cycle. The internal state should be
reset to zero when the reset input is one. Assume all values are encoded
as two's complement binary numbers. Assume all sequential logic is
triggered on the positive edge of the clock.

Enclose your code with ```verilog and ```. Only output the code snippet
and do NOT output anything else.

Answer:
""",
            },
            {
                "role": "assistant",
                "content": """```verilog
module TopModule
(
  input  logic       clk,
  input  logic       reset,
  input  logic [7:0] in_,
  output logic [7:0] out
);

  // Sequential logic

  logic [7:0] reg_out;

  always @( posedge clk ) begin
    if ( reset )
      reg_out <= 0;
    else
      reg_out <= in_;
  end

  // Combinational logic

  logic [7:0] temp_wire;

  always @(*) begin
    temp_wire = reg_out + 1;
  end

  // Structural connections

  assign out = temp_wire;

endmodule
```
""",
            },
        ]

    def _few_shot_3(self):
        return [
            {
                "role": "uesr",
                "content": """Question:
Implement a hardware module named TopModule with the following interface.
All input and output ports are one bit unless otherwise specified.

 - input  clk
 - input  reset
 - input  in_
 - output out

The module should implement a finite-state machine that takes as input a
serial bit stream and outputs a one whenever the bit stream contains two
consecutive one's. The output is one on the cycle _after_ there are two
consecutive one's.

Assume all sequential logic is triggered on the positive edge of the
clock. The reset input is active high synchronous and should reset the
finite-state machine to an appropriate initial state.

Enclose your code with ```verilog and ```. Only output the code snippet
and do NOT output anything else.

Answer:
""",
            },
            {
                "role": "assistant",
                "content": """```verilog
module TopModule
(
  input  logic clk,
  input  logic reset,
  input  logic in_,
  output logic out
);

  // State enum

  localparam STATE_A = 2'b00;
  localparam STATE_B = 2'b01;
  localparam STATE_C = 2'b10;

  // State register

  logic [1:0] state;
  logic [1:0] state_next;

  always @(posedge clk) begin
    if ( reset )
      state <= STATE_A;
    else
      state <= state_next;
  end

  // Next state combinational logic

  always @(*) begin
    state_next = state;
    case ( state )
      STATE_A: state_next = ( in_ ) ? STATE_B : STATE_A;
      STATE_B: state_next = ( in_ ) ? STATE_C : STATE_A;
      STATE_C: state_next = ( in_ ) ? STATE_C : STATE_A;
    endcase
  end

  // Output combinational logic

  always @(*) begin
    out = 1'b0;
    case ( state )
      STATE_A: out = 1'b0;
      STATE_B: out = 1'b0;
      STATE_C: out = 1'b1;
    endcase
  end

endmodule
```
""",
            },
        ]

    def _few_shot_4(self):
        return [
            {
                "role": "uesr",
                "content": """Question:
Implement a hardware module named TopModule with the following interface.
All input and output ports are one bit unless otherwise specified.

 - input  in0
 - input  in1
 - output out

The module should implement a XOR gate.

Enclose your code with ```verilog and ```. Only output the code snippet
and do NOT output anything else.

Answer:
""",
            },
            {
                "role": "assistant",
                "content": """```verilog
module TopModule
(
  input  logic in0,
  input  logic in1,
  output logic out
);

  assign out = in0 ^ in1;

endmodule
```
""",
            },
        ]

    def collate_fn(self, batch):
        dataset = [item["dataset"] for item in batch]
        problem_names = [item["problem_name"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        refs = [item["ref"] for item in batch]

        inputs = []
        for ds, p, r in zip(dataset, prompts, refs):
            if ds == "dataset_code-complete-iccad2023":
                input = p
            elif ds == "dataset_spec-to-rtl":
                input = self._apply_prompt(p, few_shot=args.few_shot)
                input = tokenizer.apply_chat_template(
                    input,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            inputs.append(input)

        return {
            "dataset": dataset,
            "problem_name": problem_names,
            "prompt": prompts,
            "ref": refs,
            "input": inputs,
        }


def collect_verilog_tasks(dir):
    dataset = []

    for dataset_name in [
        # "dataset_code-complete-iccad2023",
        "dataset_spec-to-rtl",
    ]:
        dataset_path = os.path.join(dir, dataset_name)

        for fname in os.listdir(dataset_path):
            if fname.endswith("_prompt.txt"):
                base_name = fname.replace("_prompt.txt", "")
                prompt_path = os.path.join(dataset_path, fname)
                ref_path = os.path.join(dataset_path, base_name + "_ref.sv")

                with open(prompt_path, "r") as f:
                    prompt = f.read()
                with open(ref_path, "r") as f:
                    ref = f.read()

                task = {
                    "dataset": dataset_name,
                    "problem_name": base_name,
                    "prompt": prompt,
                    "ref": ref,
                }
                dataset.append(task)

    return dataset


def extract_verilog_task(output):
    patter = r"```verilog(.*?)```"
    code = re.findall(patter, output, re.DOTALL)
    if not code:
        return output
    else:
        # return "[BEGIN]" + code[0].strip() + "[DONE]"
        return code[0].strip()


budgets = args.budgets
build_dir = args.build_dir
verilog_tasks = collect_verilog_tasks("./verilog-eval")
dataset = VerilogTaskDataset(verilog_tasks)
dataloadr = DataLoader(
    dataset,
    batch_size=156,
    shuffle=False,
    collate_fn=dataset.collate_fn,
)

# 建立 LLM 和取樣參數
llm = LLM(
    model=model_path,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=2048,
    repetition_penalty=1.05,
)

for i in range(budgets):
    print(f"Generating samples for budget {i + 1}...")
    for batch in dataloadr:
        inputs = batch["input"]
        problem_names = batch["problem_name"]

        outputs = llm.generate(inputs, sampling_params=sampling_params)
        raw = [output.outputs[0].text for output in outputs]
        verilog_code = [
            extract_verilog_task(output.outputs[0].text) for output in outputs
        ]

        for r, v_code, problem in zip(raw, verilog_code, problem_names):
            if i + 1 < 10:
                idx = format(i + 1, "02d")
            else:
                idx = str(i + 1)

            save_dir = f"{build_dir}/{problem}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # with open(f"{save_dir}/{problem}_sample{idx}_response.txt", "w") as f:
            #     f.write(v_code)

            with open(f"{save_dir}/{problem}_sample{idx}.sv", "w") as f:
                f.write(v_code)

            with open(f"{save_dir}/{problem}_sample{idx}_raw.txt", "w") as f:
                f.write(r)

            print(f"Generated code for {problem} saved.")
