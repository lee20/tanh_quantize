
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression




# 本文件主要讨论了分组线性量化来拟合不同非线性函数的可行性
# func exp sigmoid函数的取值范围是[-6,6]
# log sqrt函数的取值范围是[0.2,6]
# 如果需要修改范围和函数，请改下面三行
input_x = np.linspace(-6,6,256*60)
def func(x):
    return np.tanh(x)
    #return 1 / (1 + np.sqrt(-x)) # sigmoid


# bit量化,bitwidth表示位宽，默认为8
def quantize_bit(bit_width = 8):
    func_accurate = func(input_x)
    func_8bit = np.round(func_accurate*math.pow(2,bit_width))/math.pow(2,bit_width)
    quantize_error = np.abs(func_accurate-func_8bit)
    return quantize_error

# 绘出error曲线
def plot_error(error):
    plt.plot(input_x,error)
    plt.show()

# 256分段线性量化
def quantize_16bit_linear():
    func_accurate = func(input_x)
    coef = np.zeros((256,2))
    quantize_coef = np.zeros((256,2))
    func_predict_before_quantize = np.zeros(256*60)
    func_predict_after_quantize = np.zeros(256*60)
    
    # 先把输入分段为256，并做线性拟合
    for i in range(256):
        linear_x = input_x[i*60:i*60+60].reshape((60,1))
        linear_y = func_accurate[i*60:i*60+60].reshape((60,1))
        lineModel = LinearRegression()
        lineModel.fit(linear_x, linear_y)
        coef[i][0] = lineModel.coef_[0][0]
        coef[i][1] = lineModel.intercept_[0]
        
        # 量化线性参数
        quantize_coef[i][1] = np.round(coef[i][1]*65536)/65536
        quantize_coef[i][0] = np.round(coef[i][0]*65536)/65536

        quant_result = (linear_x.reshape(60))*quantize_coef[i][0] + quantize_coef[i][1]
        func_predict_after_quantize[i*60:i*60+60] = quant_result
    
    error_absolute = np.abs(func_accurate-func_predict_after_quantize)
    error_relative = np.abs(func_accurate-func_predict_after_quantize)/np.abs(func_accurate)
    quantize_error = np.minimum(error_absolute,error_relative)

    return quantize_error
    
    
if __name__ == "__main__":

    error = quantize_16bit_linear()
    plot_error(error)
    