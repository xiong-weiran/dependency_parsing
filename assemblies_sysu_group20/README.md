# 在google_test文件夹中运行以下命令（cd google_test）
mkdir -p build
cd build
cmake ..
make
./parser_project

# 性能测试的功能已包含在src文件夹下我们的源代码中，通过运行即可完成性能测试。
# 在src文件夹中运行以下命令（cd src）
mkdir -p build
cd build
cmake ..
make
./project
