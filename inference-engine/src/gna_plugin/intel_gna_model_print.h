#pragma once
/*

INTEL CONFIDENTIAL

Copyright 2020 Intel Corporation All Rights Reserved.

The source code contained or described herein and all documents related
to the source code("Material") are owned by Intel Corporation or its suppliers or licensors.
Title to the Material remains with Intel Corporation or its suppliers and licensors.
The Material contains trade secrets and proprietary and confidential information
of Intel or its suppliers and licensors.The Material is protected by worldwide copyright
and trade secret laws and treaty provisions.No part of the Material may be used, copied,
reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed
in any way without Intel's prior express written permission.

No license under any patent, copyright, trade secret or other intellectual property right
is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly,
by implication, inducement, estoppel or otherwise.Any license under such intellectual property
rights must be express and approved by Intel in writing.
*/

#include <iostream>
#include <gna2-model-api.h>
#include "intel_gna_model_print.h"
//#include "intel_gna_utils.h"

/**
 *      This header file contains macros that print DNN and GNA structures to human readable output.
 *      These functions are intended to be used during debugging.
 *      GNA 2.0 Compatible!
 */
#define SHOW_STRUCTURE_FIELD(stream, struct_name, field_name) do{(stream) <<                           \
                             /*'(' << typeid(struct_name).name() << ") " <<*/                          \
                             #struct_name "::" #field_name " = " << struct_name.field_name <<          \
                             /*" [" << typeid(struct_name.field_name).name() << ']' << */ '\n';}while(0)


static inline void PrintShape(std::ostream& str, const Gna2Shape* shape)
{
    if (!shape)
    {
        str << "[NULL]";
        return;
    }

    str << '[';
    for (uint32_t idx = 0; idx < shape->NumberOfDimensions && idx < GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS; ++idx)
    {
        str << shape->Dimensions[idx];
        if (idx < shape->NumberOfDimensions - 1)
        {
            str << 'x';
        }
    }
    if (shape->NumberOfDimensions > GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS)
    {
        str << "... too much dims: " << shape->NumberOfDimensions;
    }
    str << ']';
}

void PrintTensorDataElementType(std::ostream& str, const Gna2Tensor& obj)
{
    switch (obj.Type)
    {
    case Gna2DataTypeNone: str << "[None]"; return;
    case Gna2DataTypeBoolean: str << "[bool]"; return;
    case Gna2DataTypeInt4: str << "[int4]"; return;
    case Gna2DataTypeInt8: str << "[int8]"; return;
    case Gna2DataTypeInt16: str << "[int16]"; return;
    case Gna2DataTypeInt32: str << "[int32]"; return;
    case Gna2DataTypeInt64: str << "[int64]"; return;
    case Gna2DataTypeUint4: str << "[uint4]"; return;
    case Gna2DataTypeUint8: str << "[uint8]"; return;
    case Gna2DataTypeUint16: str << "[uint16]"; return;
    case Gna2DataTypeUint32: str << "[uint32]"; return;
    case Gna2DataTypeUint64: str << "[uint64]"; return;
    case Gna2DataTypeCompoundBias: str << "[COMPOUND BIAS]"; return;
    case Gna2DataTypePwlSegment: str << "[PWL seg]"; return;
    case Gna2DataTypeWeightScaleFactor: str << "[weight scale factor]"; return;
    default:
        str << "[(unknown id=" << obj.Type << ")]";
    }
}

void PrintTensorType(std::ostream& str, const Gna2Tensor& obj)
{
    switch (obj.Mode)
    {
    case Gna2TensorModeDefault:
        str << "Memory";
        if (!obj.Data) { str << "(NULL!)"; }
        break;
    case Gna2TensorModeConstantScalar:
        str << "Scalar";
        if (!obj.Data) { str << "(NULL!)"; }
        break;
    case Gna2TensorModeDisabled:
        str << "Disabled";
        break;
    default:
        str << "Unknown mode! (" << obj.Mode << ')';
    }
}

static inline void PrintTensor(std::ostream& str, const Gna2Tensor *obj, const char* desc)
{
    str << desc << ": ";

    if (nullptr == obj)
    {
        str << "[NULL]\n";
        return;
    }
    PrintShape(str, &obj->Shape);
    str << " ";
    PrintTensorType(str, *obj);
    str << " ";
    PrintTensorDataElementType(str, *obj);
    str << '\n';
}

const Gna2Tensor * getOper(const Gna2Operation& operation, uint32_t idx)
{
    return ((idx < operation.NumberOfOperands) ? operation.Operands[idx] : nullptr);
}

const void * getParam(const Gna2Operation& operation, uint32_t idx)
{
    return ((idx < operation.NumberOfParameters) ? operation.Parameters[idx] : nullptr);
}

void PrintHeader(std::ostream& str, const char* name, const Gna2Operation& operation)
{
    str << name << ", Operands: " << operation.NumberOfOperands << ", Params: " << operation.NumberOfParameters << '\n';
}

void PrintOperationConv(std::ostream& str, const Gna2Operation& operation)
{
    PrintHeader(str, "CONVOLUTION", operation);
    PrintTensor(str, getOper(operation, 0), "Input");
    PrintTensor(str, getOper(operation, 1), "Output");
    PrintTensor(str, getOper(operation, 2), "Kernels");
    PrintTensor(str, getOper(operation, 3), "Biases");
    PrintTensor(str, getOper(operation, 4), "pwl");

    str << "In stride: ";
    PrintShape(str, reinterpret_cast<const Gna2Shape*>(getParam(operation, 0)));
    str << '\n';

    if (getParam(operation, 1))
    {
        str << "BiasMode: " << *reinterpret_cast<const Gna2BiasMode*>(getParam(operation, 1)) << '\n';
    }

    if (getParam(operation, 2))
    {
        str << "PoolMode: " << *reinterpret_cast<const Gna2PoolingMode*>(getParam(operation, 2)) << '\n';
    }

    if (getParam(operation, 3))
    {
        str << "Pool Window: ";
        PrintShape(str, reinterpret_cast<const Gna2Shape*>(getParam(operation, 3)));
        str << '\n';
    }

    if (getParam(operation, 4))
    {
        str << "Pool Stride: ";
        PrintShape(str, reinterpret_cast<const Gna2Shape*>(getParam(operation, 4)));
        str << '\n';
    }

    if (getParam(operation, 5))
    {
        str << "Zero padding: ";
        PrintShape(str, reinterpret_cast<const Gna2Shape*>(getParam(operation, 5)));
        str << '\n';
    }
};

void PrintOperationCopy(std::ostream& str, const Gna2Operation& operation)
{
    PrintHeader(str, "COPY", operation);
    PrintTensor(str, getOper(operation, 0), "Input");
    PrintTensor(str, getOper(operation, 1), "Output");

    str << "Shape: ";
    PrintShape(str, reinterpret_cast<const Gna2Shape*>(getParam(operation, 0)));
    str << '\n';
};

void PrintOperationDiagTransposition(std::ostream& str, const Gna2Operation& operation)
{
    PrintHeader(str, "TRANSPOSITION", operation);
    PrintTensor(str, getOper(operation, 0), "Input");
    PrintTensor(str, getOper(operation, 1), "Output");
};

void PrintOperationFullyConnectedAffine(std::ostream& str, const Gna2Operation& operation)
{
    PrintHeader(str, "FULL AFFINE", operation);
    PrintTensor(str, getOper(operation, 0), "Input");
    PrintTensor(str, getOper(operation, 1), "Output");
    PrintTensor(str, getOper(operation, 2), "Weights");
    PrintTensor(str, getOper(operation, 3), "Biases");
    PrintTensor(str, getOper(operation, 4), "pwl");
    PrintTensor(str, getOper(operation, 5), "Weight Scale Factors");

    if (getParam(operation, 0))
    {
        str << "BiasMode: " << *reinterpret_cast<const Gna2BiasMode*>(getParam(operation, 0)) << '\n';
    }

    if (getParam(operation, 1))
    {
        str << "BiasMode: " << *reinterpret_cast<const uint32_t*>(getParam(operation, 1)) << '\n';
    }
};

void PrintOperationDiagAffine(std::ostream& str, const Gna2Operation& operation)
{
    PrintHeader(str, "DIAG AFFINE", operation);
    PrintTensor(str, getOper(operation, 0), "Input");
    PrintTensor(str, getOper(operation, 1), "Output");
    PrintTensor(str, getOper(operation, 2), "Weights");
    PrintTensor(str, getOper(operation, 3), "Biases");
    PrintTensor(str, getOper(operation, 4), "pwl");
};

static inline void PrintOperation(std::ostream& str, const Gna2Operation& operation)
{
    switch (operation.Type)
    {
    case Gna2OperationTypeConvolution:
        PrintOperationConv(str, operation);
        break;
    case Gna2OperationTypeCopy:
        PrintOperationCopy(str, operation);
        break;
    case Gna2OperationTypeFullyConnectedAffine:
        PrintOperationFullyConnectedAffine(str, operation);
        break;
    case Gna2OperationTypeElementWiseAffine:
        PrintOperationDiagAffine(str, operation);
        break;
    case Gna2OperationTypeTransposition:
        PrintOperationDiagTransposition(str, operation);
        break;
    default:
        str << "Operation: " << operation.Type << " is not defined" << '\n';
    }
}

inline void PrintModel(std::ostream& str, const Gna2Model &model)
{
    SHOW_STRUCTURE_FIELD(str, model, NumberOfOperations);
    for (unsigned int i = 0; i < model.NumberOfOperations; ++i)
    {
        str << "=== Printing operation idx=" << i << ": ===" << '\n';
        PrintOperation(str, model.Operations[i]);
    }
}
