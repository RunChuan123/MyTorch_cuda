#pragma once
#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>



class Tensor;
class TensorStorage;


enum class OpType{
    None,
    Add,
    Mul,
    MatMul,
    Relu,
    Sub,
    Div,
    Pow,
};


// 每当进行一次运算backwardfn对应的forwardfn
// 就会生成一个新的tensor，绑定到tensornode上
class TensorNode{
public:
    // 依赖的张量 前驱节点
    std::vector<Tensor*> parents;
    int version = 0;
    OpType op = OpType::None;
    // 梯度函数 
    std::function<void()> backward_fn;

    bool is_leaf= false;
    Tensor* owner = nullptr;  // 当前节点对应的Tensor

    std::unordered_map<std::string,std::vector<int>> int_args;
    std::unordered_map<std::string,long long > ll_args;
    std::unordered_map<std::string,float> float_args;
    std::unordered_map<std::string,bool> bool_args;

    std::unordered_map<std::string,std::weak_ptr<TensorStorage>> tensor_args;

    TensorNode() = default;
    TensorNode(Tensor* t) : owner(t) {}
};


