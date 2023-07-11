#include <iostream>
#include <fstream>
#include <string> 

using namespace std;

int main() {
    ofstream out("output.c");
    out << "#include <stdio.h>\nint main() {\n\t";
    out << "\n\treturn 0;\n}";
    out.close();
}