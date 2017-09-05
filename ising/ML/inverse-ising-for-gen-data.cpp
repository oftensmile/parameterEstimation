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

inline double std_rand()
{
    return rand() / (RAND_MAX + 1.0);
}

int L = 10;
int N = L*L;                          // number of spins in x and y
int n_q = 2;                          // number of state 

double J_p = 1.0;                   // spin-spin coupling constant
double T;                           // temperature in units of k_B/J_0
vector < vector<int> > S( L, vector<int>(L,0) );  
vector < vector<int> > h( L, vector<int>(L,0) );  
vector < vector<double> > test_Jx( L, vector<double>(L,0) ); 
vector < vector<double> > test_Jy( L, vector<double>(L,0) ); 
vector < vector<double> > hat_Jx( L, vector<double>(L,0) ); 
vector < vector<double> > hat_Jy( L, vector<double>(L,0) ); 

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
		S[i][j] = int( 2*(rand()%2) - 1 );
	}}
}

void init_h(){
	for(int i=0;i<L;++i){
		for(int j=0;j<L;++j){
		h[i][j] =  1;
	}}
} 

// this test_J should be printed as a dat-file.
void set_test_J(){
	for(int x=0;x<L;x++){
	for(int y=0;y<L;y++){
		// This is coupling constant is same with the case of normal potts model.			
			//test_Jx[x][y] = std_rand();	
			//test_Jy[x][y] = std_rand();	
			test_Jx[x][y] =1.0 ;	
			test_Jy[x][y] =1.0 ;	
		}}

	string fname = "./J_test-L"+to_string(L)+"-T"+to_string_with_precision(T)+"-MCMC.dat";
	ofstream file(fname);
	for(int x=0;x<L;x++){
	for(int y=0;y<L;y++){
		file<<test_Jx[x][y]<<endl;
	
	}}
	for(int x=0;x<L;x++){
	for(int y=0;y<L;y++){
		file<<test_Jy[x][y]<<endl;
	}}
	file.close();
}

double E_gene(int x, int y)
{
    int x_plus = x + 1, x_minus = x - 1, y_plus = y + 1, y_minus = y - 1;
    if (x_plus  ==  L)  x_plus  = 0;
    if (x_minus == -1)  x_minus = L - 1;
    if (y_plus  ==  L)  y_plus  = 0;
    if (y_minus == -1)  y_minus = L - 1;

    // Suppose short range interaction., matrix erements is the clock model like. 

    double E = - S[x][y] * (
         test_Jx[x][y]*S[x_plus][y] + test_Jx[x_minus][y]*S[x_minus][y]
       + test_Jy[x][y]*S[x][y_plus] + test_Jy[x][y_minus]*S[x][y_minus] );
    return E;
}

double E()
{
    // total energy
    double E_sum = 0;
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y)
            //E_sum += E(x, y);
            E_sum += E_gene(x, y);
    return E_sum / 2;
}

//  m := q/(q-1) <(S_i -1/q)>, this formulation is correspond to the case of Ising(q=2).
double M(){
	double m=0;
	for(int x=0;x<L;++x){
		for(int y=0;y<L;++y){
		m += S[x][y];	
	}}
	return m;
}

bool metropolis_step(           // true/false if step accepted/rejected
    int x, int y,               // change spin at size x,y
    double delta = 1.0 )	// maximum step size in radians 
{
    // find the local energy at site (x,y)
    // double E_xy = E(x, y);
    double E_xy = E_gene(x, y);

    // save the value of spin at (x,y)
    double S_save = S[x][y];

    // trial Metropolis step for spin at site (x,y)
    S[x][y] *= -1;

    //double E_xy_trial = E(x, y);
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

void print_state(int step){
	string fname = "state-L"+to_string(L)+"-T"+to_string_with_precision(T)+"-MCMC"+to_string(step)+".dat";
	ofstream file(fname);
	for(int i=0;i<L;++i){
		for(int j=0;j<L;++j){
		file<<S[i][j]<<" ";
		} file<<endl;
	}
}

int main()
{
    cout << " Monte Carlo Simulation of the 2-dimensional Ising Model\n"
         << " ----------------------------------------------------\n"
         << " " << L*L << " spins on " << L << "x" << L << " lattice,"<<endl;
    init_S();
    //init_h(); // introduce external field.

    T = 2.0;
    //int equilibration_steps = 200000;
    //int production_steps = 200000;
    int equilibration_steps = 2000;
    int production_steps = 2000;
    int np = 200; 
    set_test_J();
    for (int step = 0; step < equilibration_steps; ++step)
        monte_carlo_sweep();

    double e_av = 0, e_sqd_av = 0;      // average E per spin, squared average
    double m_av = 0, m_sqd_av = 0;      // average E per spin, squared average
    double accept_percent = 0;
    for (int step = 0; step < production_steps; ++step) {
        double num_spins = N;
    	if(step%np == 0){
	print_state(step);
        accept_percent += monte_carlo_sweep() / num_spins;
        double e_step = E() / num_spins;
        double m_step = M() / num_spins;
        e_av += e_step;
        e_sqd_av += e_step * e_step;
        m_av += m_step;
        m_sqd_av += m_step * m_step;
	}
    }
    cout << " done" << endl;
    double per_step= double(np) / double(production_steps);
    accept_percent *= per_step * 100.0;
    e_av *= per_step;
    e_sqd_av *= per_step;
    m_av *= per_step;
    m_sqd_av *= per_step;
    double c = (e_sqd_av - e_av * e_av) / (T * T);
    double chi = (m_sqd_av - m_av * m_av) / T;
    cout << " T = " << T << '\t'
         << " <e> = " << e_av << '\t'
         << " <e^2> = " << e_sqd_av << '\t'
         << " <m> = " << m_av << '\t'
         << " <m^2> = " << m_sqd_av << '\t'
         << " c = " << c << '\t'
         << " chi = " << chi << '\t'
         << " %accepts = " << accept_percent << endl;
    return 0;
}
