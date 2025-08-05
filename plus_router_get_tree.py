import json
import tqdm
from transformers import AutoTokenizer
import json, random, tqdm, argparse, os


tree_str = """\
Title --> Abstract --> Algorithm Innovation --> Problem --> Task Description --> Task Flow<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Problem --> Task Description --> Task Flow **
Title --> Abstract --> Algorithm Innovation --> Problem --> Task Description --> Research Value<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Problem --> Task Description --> Research Value **
Title --> Abstract --> Algorithm Innovation --> Problem --> Task Description<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Problem --> Work Motivation --> Existing Defects<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Problem --> Work Motivation --> Existing Defects **
Title --> Abstract --> Algorithm Innovation --> Problem --> Work Motivation --> Algorithm Objectives<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Problem --> Work Motivation --> Algorithm Objectives **
Title --> Abstract --> Algorithm Innovation --> Problem --> Work Motivation<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Problem<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method --> Core Innovation --> Inspiration Source<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method --> Core Innovation --> Inspiration Source **
Title --> Abstract --> Algorithm Innovation --> Method --> Core Innovation --> Core Improvement<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method --> Core Innovation --> Core Improvement **
Title --> Abstract --> Algorithm Innovation --> Method --> Core Innovation<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method --> Implementation Details --> Algorithm Components<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method --> Implementation Details --> Algorithm Components **
Title --> Abstract --> Algorithm Innovation --> Method --> Implementation Details --> Implementation Process<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method --> Implementation Details --> Implementation Process **
Title --> Abstract --> Algorithm Innovation --> Method --> Implementation Details<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Method<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment --> Experimental Design --> Baseline Methods<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment --> Experimental Design --> Baseline Methods **
Title --> Abstract --> Algorithm Innovation --> Experiment --> Experimental Design --> Experimental Datasets<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment --> Experimental Design --> Experimental Datasets **
Title --> Abstract --> Algorithm Innovation --> Experiment --> Experimental Design<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment --> Result Analysis --> Main Conclusions<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment --> Result Analysis --> Main Conclusions **
Title --> Abstract --> Algorithm Innovation --> Experiment --> Result Analysis --> Analytical Conclusions<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment --> Result Analysis --> Analytical Conclusions **
Title --> Abstract --> Algorithm Innovation --> Experiment --> Result Analysis<|im_end|>
Title --> Abstract --> Algorithm Innovation --> Experiment<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem --> Task Description --> Task Flow<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem --> Task Description --> Task Flow **
Title --> Abstract --> Benchmark Construction --> Problem --> Task Description --> Research Value<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem --> Task Description --> Research Value **
Title --> Abstract --> Benchmark Construction --> Problem --> Task Description<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem --> Motivation --> Existing Defects<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem --> Motivation --> Existing Defects **
Title --> Abstract --> Benchmark Construction --> Problem --> Motivation --> Benchmark Objectives<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem --> Motivation --> Benchmark Objectives **
Title --> Abstract --> Benchmark Construction --> Problem --> Motivation<|im_end|>
Title --> Abstract --> Benchmark Construction --> Problem<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method --> Construction Method --> Data Sources<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method --> Construction Method --> Data Sources **
Title --> Abstract --> Benchmark Construction --> Method --> Construction Method --> Annotation Scheme<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method --> Construction Method --> Annotation Scheme **
Title --> Abstract --> Benchmark Construction --> Method --> Construction Method<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method --> Evaluation System --> Evaluation Dimensions<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method --> Evaluation System --> Evaluation Dimensions **
Title --> Abstract --> Benchmark Construction --> Method --> Evaluation System --> Evaluation Metrics<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method --> Evaluation System --> Evaluation Metrics **
Title --> Abstract --> Benchmark Construction --> Method --> Evaluation System<|im_end|>
Title --> Abstract --> Benchmark Construction --> Method<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment --> Experimental Design --> Selected Baselines<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment --> Experimental Design --> Selected Baselines **
Title --> Abstract --> Benchmark Construction --> Experiment --> Experimental Design --> Experimental Settings<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment --> Experimental Design --> Experimental Settings **
Title --> Abstract --> Benchmark Construction --> Experiment --> Experimental Design<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment --> Result Analysis --> Main Conclusions<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment --> Result Analysis --> Main Conclusions **
Title --> Abstract --> Benchmark Construction --> Experiment --> Result Analysis --> Analytical Conclusions<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment --> Result Analysis --> Analytical Conclusions **
Title --> Abstract --> Benchmark Construction --> Experiment --> Result Analysis<|im_end|>
Title --> Abstract --> Benchmark Construction --> Experiment<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem --> Research Subject --> Scientific Problem<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem --> Research Subject --> Scientific Problem **
Title --> Abstract --> Mechanism Exploration --> Problem --> Research Subject --> Research Value<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem --> Research Subject --> Research Value **
Title --> Abstract --> Mechanism Exploration --> Problem --> Research Subject<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem --> Motivation --> Existing Defects<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem --> Motivation --> Existing Defects **
Title --> Abstract --> Mechanism Exploration --> Problem --> Motivation --> Exploration Objectives<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem --> Motivation --> Exploration Objectives **
Title --> Abstract --> Mechanism Exploration --> Problem --> Motivation<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Problem<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method --> Practical Exploration --> Used Components<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method --> Practical Exploration --> Used Components **
Title --> Abstract --> Mechanism Exploration --> Method --> Practical Exploration --> Implementation Process<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method --> Practical Exploration --> Implementation Process **
Title --> Abstract --> Mechanism Exploration --> Method --> Practical Exploration<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method --> Theoretical Derivation --> Fundamental Theories<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method --> Theoretical Derivation --> Fundamental Theories **
Title --> Abstract --> Mechanism Exploration --> Method --> Theoretical Derivation --> Derivation Process<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method --> Theoretical Derivation --> Derivation Process **
Title --> Abstract --> Mechanism Exploration --> Method --> Theoretical Derivation<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Method<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment --> Conclusion Presentation --> Exploration Conclusions<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment --> Conclusion Presentation --> Exploration Conclusions **
Title --> Abstract --> Mechanism Exploration --> Experiment --> Conclusion Presentation --> Supporting Evidence<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment --> Conclusion Presentation --> Supporting Evidence **
Title --> Abstract --> Mechanism Exploration --> Experiment --> Conclusion Presentation<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment --> Guidance --> Improvement Directions<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment --> Guidance --> Improvement Directions **
Title --> Abstract --> Mechanism Exploration --> Experiment --> Guidance --> Other Discussions<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment --> Guidance --> Other Discussions **
Title --> Abstract --> Mechanism Exploration --> Experiment --> Guidance<|im_end|>
Title --> Abstract --> Mechanism Exploration --> Experiment<|im_end|>
Title --> Abstract --> Survey Review --> Overview --> Field Status --> Research Scope<|im_end|>
Title --> Abstract --> Survey Review --> Overview --> Field Status --> Research Scope **
Title --> Abstract --> Survey Review --> Overview --> Field Status --> Development Context<|im_end|>
Title --> Abstract --> Survey Review --> Overview --> Field Status --> Development Context **
Title --> Abstract --> Survey Review --> Overview --> Field Status<|im_end|>
Title --> Abstract --> Survey Review --> Overview --> Motivation --> Existing Limitations<|im_end|>
Title --> Abstract --> Survey Review --> Overview --> Motivation --> Existing Limitations **
Title --> Abstract --> Survey Review --> Overview --> Motivation --> Review Objectives<|im_end|>
Title --> Abstract --> Survey Review --> Overview --> Motivation --> Review Objectives **
Title --> Abstract --> Survey Review --> Overview --> Motivation<|im_end|>
Title --> Abstract --> Survey Review --> Overview<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy --> Classification System --> Classification Dimensions<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy --> Classification System --> Classification Dimensions **
Title --> Abstract --> Survey Review --> Taxonomy --> Classification System --> Typical Methods<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy --> Classification System --> Typical Methods **
Title --> Abstract --> Survey Review --> Taxonomy --> Classification System<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy --> Key Issue Analysis --> Technical Challenges<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy --> Key Issue Analysis --> Technical Challenges **
Title --> Abstract --> Survey Review --> Taxonomy --> Key Issue Analysis --> Methodological Limitations<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy --> Key Issue Analysis --> Methodological Limitations **
Title --> Abstract --> Survey Review --> Taxonomy --> Key Issue Analysis<|im_end|>
Title --> Abstract --> Survey Review --> Taxonomy<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions --> Trend Prediction --> Evolution Path<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions --> Trend Prediction --> Evolution Path **
Title --> Abstract --> Survey Review --> Future Directions --> Trend Prediction --> New Opportunities<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions --> Trend Prediction --> New Opportunities **
Title --> Abstract --> Survey Review --> Future Directions --> Trend Prediction<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions --> Research Recommendations --> Breakthrough Directions<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions --> Research Recommendations --> Breakthrough Directions **
Title --> Abstract --> Survey Review --> Future Directions --> Research Recommendations --> Evaluation Framework<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions --> Research Recommendations --> Evaluation Framework **
Title --> Abstract --> Survey Review --> Future Directions --> Research Recommendations<|im_end|>
Title --> Abstract --> Survey Review --> Future Directions<|im_end|>
Title --> Abstract --> Theory Proof --> Problem --> Theoretical Problem --> Mathematical Formulation<|im_end|>
Title --> Abstract --> Theory Proof --> Problem --> Theoretical Problem --> Mathematical Formulation **
Title --> Abstract --> Theory Proof --> Problem --> Theoretical Problem --> Research Value<|im_end|>
Title --> Abstract --> Theory Proof --> Problem --> Theoretical Problem --> Research Value **
Title --> Abstract --> Theory Proof --> Problem --> Theoretical Problem<|im_end|>
Title --> Abstract --> Theory Proof --> Problem --> Motivation --> Theoretical Deficiencies<|im_end|>
Title --> Abstract --> Theory Proof --> Problem --> Motivation --> Theoretical Deficiencies **
Title --> Abstract --> Theory Proof --> Problem --> Motivation --> Proof Objectives<|im_end|>
Title --> Abstract --> Theory Proof --> Problem --> Motivation --> Proof Objectives **
Title --> Abstract --> Theory Proof --> Problem --> Motivation<|im_end|>
Title --> Abstract --> Theory Proof --> Problem<|im_end|>
Title --> Abstract --> Theory Proof --> Framework --> Theoretical Construction --> Fundamental Axioms<|im_end|>
Title --> Abstract --> Theory Proof --> Framework --> Theoretical Construction --> Fundamental Axioms **
Title --> Abstract --> Theory Proof --> Framework --> Theoretical Construction --> Conceptual System<|im_end|>
Title --> Abstract --> Theory Proof --> Framework --> Theoretical Construction --> Conceptual System **
Title --> Abstract --> Theory Proof --> Framework --> Theoretical Construction<|im_end|>
Title --> Abstract --> Theory Proof --> Framework --> Proof Techniques --> Proof Tools<|im_end|>
Title --> Abstract --> Theory Proof --> Framework --> Proof Techniques --> Proof Tools **
Title --> Abstract --> Theory Proof --> Framework --> Proof Techniques --> Innovative Methods<|im_end|>
Title --> Abstract --> Theory Proof --> Framework --> Proof Techniques --> Innovative Methods **
Title --> Abstract --> Theory Proof --> Framework --> Proof Techniques<|im_end|>
Title --> Abstract --> Theory Proof --> Framework<|im_end|>
Title --> Abstract --> Theory Proof --> Validation --> Rigorous Proof --> Proof Process<|im_end|>
Title --> Abstract --> Theory Proof --> Validation --> Rigorous Proof --> Proof Process **
Title --> Abstract --> Theory Proof --> Validation --> Rigorous Proof --> Boundary Conditions<|im_end|>
Title --> Abstract --> Theory Proof --> Validation --> Rigorous Proof --> Boundary Conditions **
Title --> Abstract --> Theory Proof --> Validation --> Rigorous Proof<|im_end|>
Title --> Abstract --> Theory Proof --> Validation --> Application Validation --> Case Validation<|im_end|>
Title --> Abstract --> Theory Proof --> Validation --> Application Validation --> Case Validation **
Title --> Abstract --> Theory Proof --> Validation --> Application Validation --> Extended Applications<|im_end|>
Title --> Abstract --> Theory Proof --> Validation --> Application Validation --> Extended Applications **
Title --> Abstract --> Theory Proof --> Validation --> Application Validation<|im_end|>
Title --> Abstract --> Theory Proof --> Validation<|im_end|>
Title --> Abstract<|im_end|>
Title<|im_end|>"""

maxlayer = 0
layercount = [0] * 2000


def getpreftokens(indexs, chunks, layer):
    global maxlayer, layercount
    print("layer", layer)
    print("len(indexs)", len(indexs))
    
    prefixs = {}
    tokenid2docid = {}
    for index in indexs:
        if chunks[index][layer] not in prefixs:
            prefixs[chunks[index][layer]] = {}
            tokenid2docid[chunks[index][layer]] = []
        tokenid2docid[chunks[index][layer]].append(index)
    for tokenid in tokenid2docid:
        if len(tokenid2docid[tokenid]) > 1:
            try:
                prefixs[tokenid] = getpreftokens(tokenid2docid[tokenid],chunks,layer + 1)
            except:
                raise ValueError("layer", layer, "tokenid", tokenid, "tokenid2docid[tokenid]", tokenid2docid[tokenid])
        else:
            prefixs[tokenid] = 0
            if layer > maxlayer:
                maxlayer = layer
            layercount[layer] += 1
    return prefixs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", "--model_path", type=str, default="/141nfs/username/hf_models/Qwen3-0.6B")
    parser.add_argument("-outfile", "--outfile", type=str, default="utils/plus_tree.json")
    args = parser.parse_args()
    print("args.model_path", args.model_path)
    print("args.outfile", args.outfile)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    chunks = []
    for path in tree_str.split("\n"):
        path = f" ** {path}"
        chunks.append(tokenizer.encode(path, add_special_tokens=False))
    prefixs = getpreftokens(list(range(len(chunks))), chunks, 0)

    print("maxlayer", maxlayer)
    print("layercount", layercount)
    json.dump(prefixs, open(args.outfile,'w'))

    print("sum(layercount)", sum(layercount))