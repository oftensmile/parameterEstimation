//  Two-dimensional Potts Model
// iAskin , Juius; Teller , Edward(1943). "Statistics of Two-Dimensional Lattices With Four Componets"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
using namespace std;

inline double std_rand(){ return rand() / (RAND_MAX + 1.0); }

int L = 10;
int N = L*L;                          // number of spins in x and y
int n_q = 3;                          // number of state 
// S_i = 0, 1,.., n_q-1
int n_sample = 10000; // iteration number of the optimaisation.
int epoc_stop = 300; // iteration number of the optimaisation.
double lr = 2.5;	// learning rate of Gradient Decent
double regula = 1/n_sample;	// 
double T=0.8;                           // temperature in units of k_B/J_0
double cost_J, cost_K;

double J_p = 1.0;                   // spin-spin coupling constant
vector < vector<int> > S( L, vector<int>(L,0) );  
vector < vector<int> > S_data( L, vector<int>(L,0) );  
vector < vector<int> > h( L, vector<int>(L,0) );  

vector < vector<double> > K_data( 2*N, vector<double>(n_q*n_q,0) ); 
vector < vector<double> > K_modl( 2*N, vector<double>(n_q*n_q,0) );
// J_modl[u][], x-direction: J_modl= i-to-j, => L*i+j, y-direction: J_modl= i-to-j, => L*i+j+L,
vector < vector<double> > J_modl( 2*N, vector<double>(n_q*n_q,1) ); 
vector < vector<double> > test_J( 2*N, vector<double>(n_q*n_q,0) ); 

template <typename T>
string to_string_with_precision(const T a_value, const int n = 5)
{
    ostringstream out;
    out << setprecision(n) << a_value;
    return out.str();
}

void init_S(){
	for(int i=0;i<L;++i){
		for(int j=0;j<L;++j){
		S[i][j] = rand()%n_q;
	}}
}

void init_J_model(){
	for(int x=0;x<L;++x){
	for(int y=0;y<L;++y){
		for(int a=0;a<n_q;++a){
			J_modl[ x*L+y ][ a*n_q+a ] = 1.0;     
		        J_modl[ x*L+y + N ][ a*n_q+a ] = 1.0;
		}
	}}
}

bool read_file(string filename){
  bool flag = true;
  ifstream myfile (filename);
  string line;
  int count = 0; int i=0,j=0; // this initialize is necessary.
  if (myfile.is_open())
  {
    while ( getline (myfile,line,' ') && (i<L-1||j<L-1))
    {
		i = int(double(count)/L);	
		j = (count%L);	
		S_data[i][j] = stoi(line);
    count += 1;
    }
    myfile.close();
    flag = true;
  }
  else{ cout << "Unable to open file" << endl;  
  	cout << filename << endl;
  }
	return flag;
}

void set_test_J(){
	for(int x=0;x<L;x++){
	for(int y=0;y<L;y++){
		for(int a=0;a<n_q;a++){
			test_J[ x*L+y ][ a*n_q+a ] = 1.0;	
			test_J[ x*L+y + N ][ a*n_q+a ] = 1.0;
		}
		}}
}


// this energy fuction shold use J_modl. 
double E_gene(int x, int y)
{
    int x_plus = (x+1)%L, x_minus = (x-1+L)%L, y_plus =(y+1)%L , y_minus = (y-1+L)%L;
    
    // Suppose short range interaction., Use J_modl not test_J. 
    double E = -J_p * (
        J_modl[ L*x+y           ] [ n_q*S[x][y] + S[x_plus][y]  ]+
        J_modl[ L*x_minus+y     ] [ n_q*S[x][y] + S[x_minus][y] ]+
        J_modl[ L*x+y + N       ] [ n_q*S[x][y] + S[x][y_plus]  ]+
        J_modl[ L*x+y_minus + N ] [ n_q*S[x][y] + S[x][y_minus] ]	);

    return E;
}

double E()
{
    // total energy
    double E_sum = 0;
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y)
            E_sum += E_gene(x, y);
    return E_sum / 2;
}
//  m := q/(q-1) <(S_i -1/q)>, this formulation is correspond to the case of Ising(q=2).
double M(){
	double m=0;
	for(int x=0;x<L;++x){
		for(int y=0;y<L;++y){
		//m += (S[x][y] - 1.0/n_q);	
		m += (S[x][y] - 1.0/n_q);	
	}}
	return m * n_q/(n_q-1);
}

int heat_prop(int state){
	return rand()%n_q;
}

bool metropolis_step(           // true/false if step accepted/rejected
    int x, int y,               // change spin at size x,y
    double delta = 1.0 )	// maximum step size in radians 
{
    // find the local energy at site (x,y)
    double E_xy = E_gene(x, y);

    // save the value of spin at (x,y)
    double S_save = S[x][y];

    S[x][y] = heat_prop(S[x][y]);

    double E_xy_trial = E_gene(x, y);

    // Metropolis test
    bool accepted = false;
    double dE = E_xy_trial - E_xy;
    double w = exp(- dE / T);
    if (w > std_rand())
        accepted = true;
    else
        S[x][y] = S_save;
    return accepted;
}

// this monte_carlo_sweep shold use J_modl. 
int monte_carlo_sweep()
{
    int acceptances = 0;
    // randomized sweep
    for (int step = 0; step < N; ++step) {
        // choose a site at random
        int x = int(L * std_rand());
        int y = int(L * std_rand());
        if (metropolis_step(x, y))
            ++acceptances;
    }
    return acceptances;
}

void calc_K_data(){
	int x_plus,y_plus;
	int S_00,S_p0,S_0p,S_m0,S_0m;
	for(int n=0;n<n_sample;++n){
		string filename="./q3DataL10T"+to_string_with_precision(T)+"/state-L"+to_string(L)+"-T"+to_string_with_precision(T)+"-MCMC"+to_string(n*200)+".dat";
		//cout<<"filename="<<filename<<endl;
		read_file(filename);
		for(int x=0;x<L;++x){	
		for(int y=0;y<L;++y){
			x_plus=(x+1)%L,y_plus=(y+1)%L; 
			// Shold be S_data not S !!! //
			S_00 = int(S_data[x][y]);	
			S_p0 = int(S_data[x_plus][y]);	
			S_0p = int(S_data[x][y_plus]);	
			
			K_data[ L*x+y   ][ n_q*S_00 + S_p0 ] += 1.0; 	
			K_data[ L*x+y+N ][ n_q*S_00 + S_0p ] += 1.0; 	
		}}	
	}// End of sample loop.	

	// Normalization.
	double percent = 1.0/double(n_sample);
	for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){
			K_data[ i*L+j     ][ a*n_q+b ] *= percent;  
			K_data[ i*L+j + N ][ a*n_q+b ] *= percent; 
		} }	
	} }	
}

void inti_K_modl(){
	for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){
			K_modl[ L*i+j     ][ n_q*a+b ] = 0;	
			K_modl[ L*i+j + N ][ n_q*a+b ] = 0;	
		}}
	}}
}

void calc_K_model(){
	init_S();
	//int col_i,row_i,col_j,row_j,a,b;
	int x_plus,y_plus;
	int S_00,S_p0,S_0p;
    	int equilibration_steps = 10000;
    	int production_steps = 20000;
    	int np = 200;
	inti_K_modl();
	for (int step = 0; step < equilibration_steps; ++step){
		monte_carlo_sweep(); // this monte_carlo_sweep should use J_modl
	}
	for (int step = 0; step < production_steps; ++step){
		monte_carlo_sweep(); // this monte_carlo_sweep should use J_modl
		if(step%np==0){
			for(int x=0;x<L;++x){
			for(int y=0;y<L;++y){
				x_plus=(x+1)%L; y_plus=(y+1)%L;
				
				S_00 = int(S[x][y]);	
				S_p0 = int(S[x_plus][y]);	
				S_0p = int(S[x][y_plus]);	
				
				K_modl[ L*x+y   ][ n_q*S_00 + S_p0 ] += 1.0; 	
				K_modl[ L*x+y+N ][ n_q*S_00 + S_0p ] += 1.0; 	
		}}	
		}
	}
	
	// Normalization.
	double percent = double(np)/double(production_steps);
	for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){
			K_modl[ L*i+j   ][ n_q*a+b ] *= percent; 	
			K_modl[ L*i+j+N ][ n_q*a+b ] *= percent; 	
		} }
	} }	
}

void update_J_modl(){
	for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){

		J_modl[ L*i+j ][ n_q*a+b ] +=  lr*( K_data[ L*i+j ][ n_q*a+b ] - K_modl[ L*i+j ][ n_q*a+b ] ); 
		J_modl[ L*i+j ][ n_q*a+b ] += - regula*J_modl[L*i+j][n_q*a+b];
		
		J_modl[ L*i+j+N ][ n_q*a+b ] += lr*( K_data[ L*i+j+N ][ n_q*a+b ] - K_modl[ L*i+j+N ][ n_q*a+b] ); 
		J_modl[ L*i+j+N ][ n_q*a+b ] += - regula*J_modl[L*i+j+N ][n_q*a+b];
	}}
	}}
}

void evaluater_of_error(int epoc){
	cost_J=0; cost_K=0; for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){
		
		cost_J+=abs(T*J_modl[L*i+j][n_q*a+b] - test_J[L*i+j][n_q*a+b]);	
		cost_J+=abs(T*J_modl[L*i+j+N][n_q*a+b]  - test_J[L*i+j+N][n_q*a+b]);	
		cost_K+=abs(K_data[ L*j+i ][ n_q*a+b ] - K_modl[ L*j+i ][ n_q*a+b ]);	
		cost_K+=abs(K_data[ L*j+i +N ][ n_q*a+b ] - K_modl[ L*j+i +N ][ n_q*a+b ]);	
		}}
	}}
	cost_J /= double(2*N * n_q*n_q);
	cost_K /= double(2*N * n_q*n_q);
	if(epoc%100==0){
	string fname = "error"+to_string(int(double(epoc)/100))+".dat";
	ofstream file(fname);
	//print interaction of x direction//
	for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){
			// T* is correction.
		file<<J_modl[L*i+j][a*n_q+b] << " "<< T * J_modl[L*i+j][a*n_q+b] -1.0/n_q << " " <<test_J[L*i+j][a*n_q+b]<< " " 
			<<K_modl[L*i+j][a*n_q+b] << " " <<K_data[L*i+j][a*n_q+b]<<endl;
		}}
		}} 
	
	//print interaction of y direction//
	for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
		for(int a=0;a<n_q;++a){
		for(int b=0;b<n_q;++b){
		file<<J_modl[L*i+j+N][a*n_q+b] << " " << T * J_modl[L*i+j+N][a*n_q+b] -1.0/n_q << " " <<test_J[L*i+j+N][a*n_q+b]<< " " 
			<<K_modl[L*i+j+N][a*n_q+b] << " " <<K_data[L*i+j+N][a*n_q+b]<<endl;
		}}
		}} 
	file.close();
	}
}

int main()
{
	set_test_J();
	calc_K_data();
	init_J_model();
	for(int epoc=0;epoc<epoc_stop;++epoc){
		/*calclation of the data.*/
		evaluater_of_error(epoc);
		cout<<epoc<<" "<<cost_J<<" "<<cost_K<<endl;
		calc_K_model();	
		update_J_modl();
		}
        return 0;
}
