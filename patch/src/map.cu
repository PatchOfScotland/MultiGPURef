#include <iostream>

typedef float funcType;

int main(int argc, char** argv){
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <array length>\n";
        exit(EXIT_FAILURE);
    } 

    unsigned int array_len = atoi(argv[1]);

    std::cout << "Running array of length " << array_len << " (" << ((array_len*2*sizeof(funcType))/1e9) <<"GB)\n";

}
