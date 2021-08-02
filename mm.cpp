/**
 * Pipeline merge sort
 * Lukas Dobis
 * xdobis01
 * PRL 2021
 */

#include <iostream>
#include <fstream>
#include <mpi.h>
#include <math.h>
#include <vector>

#define MAT_FILE_1 "mat1" /// Name of the input file with first matrix.
#define MAT_FILE_2 "mat2" /// Name of the input file with second matrix.

#define MAIN_PROC 0	      /// Rank of main process
#define TAG_LEFT 0        /// MPI tag for messages comming with values from first matrix, along row axis.
#define TAG_UP 1          /// MPI tag for messages comming with values from second matrix, along column axis.
#define TAG_END 2         /// MPI tag for final computed values send by processors to main process.

#define TAG_M 3			  /// MPI tag for initialization message with first matrix row dimension from main process.
#define TAG_N 4			  /// MPI tag for initialization message with first matrix columns dimension from main process.
#define TAG_K 5			  /// MPI tag for initialization message with second matrix columns dimension from main process.

//#define COMPLEXITY_TEST /// Define for complexity tests.

#ifdef COMPLEXITY_TEST
#include <chrono>
#define TAG_TIME 6			  /// MPI tag for message signalling end of algorithm, from last process to main process.
#endif

using namespace std;

/**
 * Structure for holding matrix dimensions and values.
 * Also has getValue function for getting matrix values by index.
 */
struct Mat 
{
	const int rows;
	const int cols;
	vector<int> values;
	
	int getValue(const int row_index, const int col_index)
	{	
		return values.at(row_index * cols + col_index);
	}
};

/**
 * Prints an error message to standard error due to an MPI error and ends the program.
 */
void MPI_error()
{
	cerr << "Error: MPI library call has failed." << endl;
	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

/**
 * Receive number using MPI Receive message tagged with comm_tag from process with sender rank.
 *
 * @param sender_rank Rank of a sending process.
 * @param comm_tag MPI communication tag.
 * @param num Variable to hold received value.
 */
void recNum(const int sender_rank, const int comm_tag, int& num)
{	
	MPI_Status status;
	if (MPI_Recv(&num, 1, MPI_INT, sender_rank, comm_tag, MPI_COMM_WORLD, &status))
	{
		MPI_error();
	}
}

/**
 * Send number using MPI Send message tagged with comm_tag to process with receiver rank.
 *
 * @param receiver_rank Rank of a receiving process.
 * @param comm_tag MPI communication tag.
 * @param num Variable to hold received value.
 */
void sendNum(const int receiver_rank, const int comm_tag, int& num)
{	
	if (MPI_Send(&num, 1, MPI_INT, receiver_rank, comm_tag, MPI_COMM_WORLD))
	{
		MPI_error();
	}
}

/**
 * Read and return matrix from filepath inputted as argument.
 *
 * @param file_name Filename of file containing matrix.
 */
Mat readMatrix(string file_name)
{	
	int dimension;
    int cols;
    int rows;
    vector<int> values;
    int value;
    
    // Open file stream
	ifstream file(file_name);
    
    // Terminate if file opening fails
    if (!file.is_open())
    {
        cerr << "Error: Could not open file: " << file_name << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    // Read dimension number from file beginning
    file >> dimension;
    
    // Read all remaining matrix values
    while (file >> value)
    {
        values.push_back(value);
    }
    
    // Close file
    file.close();
    
    // Based on filename determine matrix order and initialize rows and columns dimension accordingly
    if (file_name == MAT_FILE_1)
    {
		rows = dimension;
		cols = values.size() / rows;
    }
    else
    {
		cols = dimension;
		rows = values.size() / cols;
    }
    
    // Initialize and return initialized matrix structure
    Mat matrix = {rows, cols, values};
    
    return matrix;
}

/**
 * Print matrix inputted as argument.
 *
 * @param matrix Matrix structure whose values are to be printed.
 */
void printMatrix(Mat matrix)
{   
	unsigned char separator;
	int value;
	
	// Print matrix dimension separated by ':' character
	cout << matrix.rows << ':' << matrix.cols << endl;
	
	// Print all matrix values
	for(int x = 0; x < matrix.rows; ++x)
    {	
        for(int y = 0; y < matrix.cols; ++y)
        {	
        	value = matrix.getValue(x,y);
			
			// Separate row values with blank space, end row with '\n' 
        	separator = y != (matrix.cols - 1) ? ' ' : '\n';
            cout<< value << separator;
        }
    }
    //cout << endl;
}

/**
 * Main processor, acts as master process that handles passing matrix values,
 * to other mesh multiplication cell processors, and also is one of them.
 * He loads both matrices and initializes other processors with matrix dimensions.
 * Then proceeds to compute his own cell position value from first row of first matrix,
 * and first column of second matrix. Each loaded value, is sent to left or bottom 
 * processor depending on value's original matrix. At end of computation iteration 
 * he loads and sends other rows/columns values for border processors of mesh multiplication 
 * processor matrix. After all processors compute their values they send their final values
 * to main processor and he prints them to standard output.
 *
 * @param procs_count Number of running processes.
 */
void mainProcessor(int procs_count)
{	
	// Cell number computed by this processor in processor matrix
	int cell_num = 0;
	// Numbers send either from left to rigth processors or from up to bottom processors
	int left_num, up_num;
	// Matrix dimensions
	int m,n,k;
	// Variables for holding ranks and for sending numbers to other border processors during computation.
	unsigned int row_rank, col_rank;
	int send_num;
	
	// Load matrices
	string filename1 = MAT_FILE_1;
	string filename2 = MAT_FILE_2;
	
	Mat mat1 = readMatrix(filename1);
	Mat mat2 = readMatrix(filename2);
	
	// Test if first matrix column dimension is equal to second matrix row dimension
	if (mat1.cols != mat2.rows)
    {
        cerr << "Error: First matrix columns dimension: " << mat1.cols << " " <<
		        "does not match second matrix rows dimension: " << mat2.rows << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
	// Test if number of active processors is equal to first matrix row dimension multiplied by second matrix column dimension
	if (procs_count != (mat1.rows * mat2.cols))
    {
        cerr << "Error: Number of used processors: " << procs_count << " " <<
		        "does not match number of expected processors: " << mat2.rows << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
	
	// Get m - first matrix rows dimension, n - shared dimension, k - second matrix column dimension
	m = mat1.rows;
	n = mat1.cols;
	k = mat2.cols;
	
	// Ranks of processors positioned lower and right of main processor position
	const unsigned int down_rank = k;
	const unsigned int right_rank = 1;
	
	// Initialize processors with messages holding matrices dimensions
    for (int recv_rank = 1; recv_rank < (m * k); ++recv_rank)
    {	
    	sendNum(recv_rank, TAG_M, m);
    	sendNum(recv_rank, TAG_N, n);
    	sendNum(recv_rank, TAG_K, k);
    }
    
	#ifdef COMPLEXITY_TEST
	// Start measuring algorithm time
	const int last_rank = (m - 1) * k + k - 1; // Rank of last process
    chrono::time_point<chrono::high_resolution_clock> start, end;
	start = chrono::high_resolution_clock::now();
    #endif
    
	// Computation cycle
	for (int i = n - 1; i >= 0; --i)
	{	 
		 // Load numbers to be passed along row and column axis
 		 left_num = mat1.getValue(0, i);
		 up_num = mat2.getValue(i, 0);

 	 	 // Send numbers along their dimensions to lower and right processors
 	 	 if (k != 1) // If second matrix is 1D column dont send number 
 	 	 {
 			 sendNum(right_rank, TAG_LEFT, left_num);
 	 	 }
  	 	 if (m != 1) // If left matrix is 1D row dont send number 
 	 	 {
 			 sendNum(down_rank, TAG_UP, up_num);
 	 	 }
		 
		 // Accumulate main processor value
 		 cell_num += left_num * up_num;
		 
  		 // Send row values of first matrix to first column of processors 
		 for (unsigned int row = 1; row < m; ++row)
		 {
		 	row_rank = row * k;
		 	send_num = mat1.getValue(row,i);
		 	sendNum(row_rank, TAG_LEFT, send_num);
		 }
 		 // Send column values of second matrix to first row of processors 
 		 for (unsigned int col = 1; col < k; ++col)
		 {
		 	col_rank = col;
		 	send_num = mat2.getValue(i,col);
		 	sendNum(col_rank, TAG_UP, send_num);
		 }
	}
    
    // Receive signal from last process and print final algorithm compute time
	#ifdef COMPLEXITY_TEST
	int empty = 0; // Placeholder value for signal message
	recNum(last_rank, TAG_TIME, empty);
	end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << "MM computational time duration: " << diff.count() << "s"<< endl;
    #endif
    
	// Initialize final matrix holding mesh multiplication computed values
	Mat matrix = {m,k};
	int num;
	
	// Insert first row and first column value from main processor
	matrix.values.push_back(cell_num);
	
	// Receive final values from all other processors
    for (unsigned int send_rank = 1; send_rank < (m * k); ++send_rank)
    {
        recNum(send_rank, TAG_END, num);
        matrix.values.push_back(num);
    }
    
	// Print computated matrix as result to standard output
	printMatrix(matrix);
}

/**
 * Cell processor of mesh multiplication algorithm, act as accumulators for values of 
 * computed matrix. They receive values from upper or left processor, if they are in 
 * first row or column they receive one of their values directly from main processor.  
 * At start they are initialized with matrix dimension messages from main processor,
 * then they start receiving numbers, which they use to compute their cell position value,
 * and send them to processors bottom or right of their own position. If they are in last,
 * row or column then one of values is not sent further. After computation ends, they 
 * send their final computed value to main processor.   
 *
 * @param rank Processor rank value.
 */
void cellProcessors(const int rank)
{	
	// Number of processed numbers
	unsigned int processed_num_counter = 0;
	
	// First and second matrices dimensions
	int m;
	int n;
	int k;
	
	// Receive initialization messages
	recNum(MAIN_PROC, TAG_M, m);
	recNum(MAIN_PROC, TAG_N, n);
	recNum(MAIN_PROC, TAG_K, k);
	
	// Set count of number to be processed
	const unsigned int num_count = n;
	
	// Determine cell processor position in mesh multiplication processor matrix 
	const unsigned int my_pos_x = rank / k;
	const unsigned int my_pos_y = rank % k; 
	
	// Determine if processor is last
	#ifdef COMPLEXITY_TEST
	bool is_last_proc = my_pos_x == (m - 1) && my_pos_y == (k - 1);
    #endif
	
	// Determine ranks of processors in all four directions
	const int up_rank = rank - k;
	const int left_rank = rank - 1;
	const unsigned int down_rank = rank + k;
	const unsigned int right_rank = rank + 1;
	
	// Cell processor value
	int cell_num = 0;
	
	// Variables for holding numbers received and passed from other processor
	int up_num, left_num;
	
	// Loop until one row of first matrix or one column of second matrix of values is processed
	while(processed_num_counter < num_count)
	{	
		// Receive value from main processor if process is in first row of processor matrix
		if (my_pos_x == 0)
		{
			recNum(MAIN_PROC, TAG_UP, up_num);
		}
		else 	// Receive value from upper processor
		{
			recNum(up_rank, TAG_UP, up_num);
		}

		// Receive value from main processor if process is in first column of processor matrix
		if (my_pos_y == 0)
		{
			recNum(MAIN_PROC, TAG_LEFT, left_num);
		}
		else	// Receive value from left processor
		{
			recNum(left_rank, TAG_LEFT, left_num);
		}

		// If cell processor is not in last row, send number from upper processor to lower processor
		if ((my_pos_x + 1) < m)
		{	
			sendNum(down_rank, TAG_UP, up_num);
		}
		// If cell processor is not in last column, send number from left processor to rigth processor
		if ((my_pos_y + 1) < k)
		{
		    sendNum(right_rank, TAG_LEFT, left_num);
		}

		// Accumulate cell processor value
		cell_num += left_num * up_num;
		
		// Last processor after computing last number, signals end of algorithm to main processor by empty message
    	#ifdef COMPLEXITY_TEST
    	if ( is_last_proc && processed_num_counter == (num_count - 1) )
    	{   
            int empty = 0;
	        sendNum(MAIN_PROC, TAG_TIME, empty);
	    }
	    #endif
		
		// Increment number of processed number pairs
		++processed_num_counter;
	}
	
	// Send final accumulated value to main processor
	sendNum(MAIN_PROC, TAG_END, cell_num);
}

// Main 
int main(int argc, char *argv[])
{	
	// MPI initialization
	MPI_Init(&argc, &argv);
	
	// Processor gets value of his rank
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Get number of used processors
	int procs_count;
	if (MPI_Comm_size(MPI_COMM_WORLD, &procs_count))
	{	
		MPI_error();
	}
	
	// Main processor 
    if (rank == MAIN_PROC)
    {	
        mainProcessor(procs_count);
    }
    else // Other processors
    {
        cellProcessors(rank);
    }
	
	MPI_Finalize();

	return EXIT_SUCCESS;
}
