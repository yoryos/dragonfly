#include <fstream>
#include <iostream>
#include <iomanip>

template<typename TYPE>
bool save_matrix(const char* path, TYPE** & matrix, int m, int n, int prec = 1) {

    std::ofstream file(path);

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            if(file) {
                file << std::fixed << std::setprecision(prec) <<  matrix[j][i];
                if( j != n - 1) file << " ";
            } else {
                std::cerr << "Error writing to: " << path << std::endl;
                file.close();
                return false;
            }
        }
        file << std::endl;
    }
    file.close();
    return true;
}

template<typename TYPE>
bool load_matrix_from_file(const char * filepath,TYPE ** &matrix,
                           int &width, int &length, char delimiter = ' ') {
    // Open file
    std::ifstream file(filepath);
    if(not file.is_open()) {
        std::cerr << "Could not open " << filepath << std::endl;
        return false;
    }

    width = 0;
    length = 0;

    char c;

    // Discover the length
    while(file.get(c)) {
        if(c == '\n') {
            ++length;
        }
        if(file.eof())
        {
            break;
        }
    }

    if(length == 0) {
        std::cerr << "File " << filepath << " is empty!" << std::endl;
        return false;
    }

    // Reset file_stream to beginning of the file
    file.clear();
    file.seekg(0, std::ios::beg);

    // Discover the width
    char c1, c2 = '\0';
    // use of c2 ensures that using the delimiter twice does not increase the width
    while(true) {
        file.get(c1);
        if((c1 == delimiter) and (c1 != c2)) {
            ++width;
        }
        if(c1 == '\n') {
            ++width;
            break;
        }
        c2 = c1;
    }

    // Reset file_stream to beginning of the file
    file.clear();
    file.seekg(0, std::ios::beg);

    try {
        matrix = new TYPE*[length];
        for(int i = 0; i < length; ++i) {
            matrix[i] = new TYPE[width];
            for(int j = 0; j < width; ++j) {
                if( not (file >> matrix[i][j])) {
                    std::cerr << "Error reading " << filepath <<std::endl;
                    return false;
                }
            }
        }
    } catch (std::bad_alloc &ba) {
        std::cerr << "Could not allocate space on the heap: " << ba.what() << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}
