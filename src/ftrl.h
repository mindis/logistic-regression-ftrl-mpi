#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "predict.h"
#include "mpi.h"
#include <math.h>

class FTRL{
    public:
        FTRL(Load_Data* load_data, Predict* predict, int total_num_proc, int my_rank) 
                        : data(load_data), pred(predict), num_proc(total_num_proc), rank(my_rank){
                                init();
        }
        ~FTRL(){}

        void init(){
            loc_w = new float[data->glo_fea_dim]();
            loc_g = new float[data->glo_fea_dim]();
            glo_g = new float[data->glo_fea_dim]();

            loc_sigma = new float[data->glo_fea_dim]();
            loc_n = new float[data->glo_fea_dim]();
            loc_z = new float[data->glo_fea_dim]();
        }

        float sigmoid(float x){
            if(x < -30) return 1e-6;
            else if(x > 30) return 1.0;
            else{
                double ex = pow(2.718281828, x);
                return ex / (1.0 + ex);
            }
        }

        void update(){// only for master node
            for(int col = 0; col < data->glo_fea_dim; col++){
                //the first update sigma, z, n
                loc_sigma[col] = ( sqrt (loc_n[col] + glo_g[col] * glo_g[col]) - sqrt(loc_n[col]) ) / alpha;
                loc_n[col] += glo_g[col] * glo_g[col];
                loc_z[col] += glo_g[col] - loc_sigma[col] * loc_w[col];
                //the secondary update w
                if(abs(loc_z[col]) <= lambda1){
                    loc_w[col] = 0.0;
                }
                else{
                    float tmpr= 0.0;
                    if(loc_z[col] >= 0) tmpr = loc_z[col] - lambda1;
                    else tmpr = loc_z[col] + lambda1;
                    float tmpl = -1 * ( ( beta + sqrt(loc_n[col]) ) / alpha  + lambda2);
                    loc_w[col] = tmpr / tmpl;
                }
            }
        }

        void batch_gradient_calculate(int &row){
            int index = 0; float value = 0.0; float pctr = 0;
            for(int line = 0; line < batch_size; line++){
                float wx = bias;
                for(int col = 0; col < data->fea_matrix[row].size(); col++){//for one instance
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    wx += loc_w[index] * value;
                }
                pctr = sigmoid(wx);
                for(int col = 0; col < data->fea_matrix[row].size(); col++){
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    loc_g[index] += (pctr - data->label[row]) * value;
                }
                row++;
            }//end for
        }

        void ftrl(){
            int batch_num = data->fea_matrix.size() / batch_size;
            int batch_num_min = 0;
            MPI_Allreduce(&batch_num, &batch_num_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            std::cout<<"epochs = "<<epochs<<" batch_num_min = "<<batch_num_min<<std::endl;
            for(int epoch = 0; epoch < epochs; epoch++){
                int row = 0, batches = 0;
                if(rank == 0) std::cout<<"epoch "<<epoch;
                pred->run(loc_w);
                while(row < data->fea_matrix.size()){
                    if( (batches == batch_num_min - 1) ) break;
                    batch_gradient_calculate(row);
                    for(int col = 0; col < data->glo_fea_dim; col++){
                        loc_g[col] /= batch_size;
                    }

                    if(rank != 0){//slave nodes send gradient to master node;
                        MPI_Send(loc_g, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
                    }
                    else if(rank == 0){//rank 0 is master node
                        for(int j = 0; j < data->glo_fea_dim; j++){//store local gradient to glo_g;
                            glo_g[j] = loc_g[j];
                        }
                        for(int r = 1; r < num_proc; r++){//receive other node`s gradient and store to glo_g;
                            MPI_Recv(loc_g, data->glo_fea_dim, MPI_FLOAT, r, 99, MPI_COMM_WORLD, &status);
                            for(int j = 0; j < data->glo_fea_dim; j++){
                                glo_g[j] += loc_g[j];
                            }
                        }
                        for(int j = 0; j < data->glo_fea_dim; j++){
                            glo_g[j] /= num_proc;
                        }
                        update();
                    }
                    //sync w of all nodes in cluster
                    if(rank == 0){
                        for(int r = 1; r < num_proc; r++){
                            MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, r, 999, MPI_COMM_WORLD);
                        }
                    }
                    else if(rank != 0){
                        MPI_Recv(loc_w, data->glo_fea_dim, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &status);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);//will it make the procedure slowly? is it necessary?
                    batches++;
                }//end row while
            }//end epoch for
        }//end ftrl

    public:
        float* loc_w;
        int epochs;
        int batch_size;
        float bias;
        float alpha;
        float beta;
        float lambda1;
        float lambda2;
    private:
        MPI_Status status;

        Load_Data* data;
        Predict* pred;
        
        float* loc_g;
        float* glo_g;
        float* loc_z;
        float* loc_sigma;
        float* loc_n;

        int num_proc;
        int rank;
};
#endif
