#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define MAX_SEQ_NUMBER 	(100)

#define MAX_THREAD_NUMBER (33)

#define MAX_PRINT_NUM	(10)

struct thread_param {
	int seq;
	int thread_num;
	pthread_mutex_t  my_thread_mutex;
	pthread_mutex_t  write_bitstream_my_mutex;
	pthread_mutex_t  *write_bitstream_next_mutex;
};



void start_encode_slice(struct thread_param *param)
{
	int ret = pthread_mutex_unlock(&param->my_thread_mutex);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	return;
}
void wait_write_bitstream(struct thread_param * param)
{
//	printf("%d %p\n", param->seq, &param->write_bitstream_next_mutex);
	int ret = pthread_mutex_lock(&param->write_bitstream_my_mutex);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	return;
}
void start_write_next_bitstream(struct thread_param * param)
{
//	printf("%d %p\n", param->seq, param->write_bitstream_next_mutex);
	if (param->write_bitstream_next_mutex != NULL ) {
		int ret = pthread_mutex_unlock(param->write_bitstream_next_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return;
		}
	}
	return;
}

void *thread_start_routin(void *arg)
{
	struct thread_param *param =  (struct thread_param*)arg;

	int counter = 0;
	for(;;) {
		//start_encode_slice(param);
		if (((counter * MAX_THREAD_NUMBER) + param->thread_num ) > MAX_SEQ_NUMBER) {
			break;
		}
		int seq = (counter * MAX_THREAD_NUMBER) + param->thread_num;
		int i;
		int num = rand();
		for (i = 0; i< num % MAX_PRINT_NUM;i++) {
			printf("dct %d %d %d\n", seq, (num %10), i);
		}
		wait_write_bitstream(param);
		num = rand();
		for (i = 0; i< num % MAX_PRINT_NUM;i++) {
			printf("mem write %d %d %d\n", seq, (num %10), i);
		}
		printf("end %d\n", seq);
		start_write_next_bitstream(param);

		counter++;
	}
	
	pthread_exit(NULL);
}

pthread_t thread[MAX_THREAD_NUMBER];

struct thread_param params[MAX_THREAD_NUMBER];


int main()
{

	int i;
	pthread_attr_t attr;
	int ret = pthread_attr_init(&attr);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return -1;
	}

	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		ret = pthread_mutex_init(&params[i].my_thread_mutex,NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}

		ret = pthread_mutex_lock(&params[i].my_thread_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		ret = pthread_mutex_init(&params[i].write_bitstream_my_mutex,NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}

		ret = pthread_mutex_lock(&params[i].write_bitstream_my_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		params[i].write_bitstream_next_mutex = &params[(i+1)%MAX_THREAD_NUMBER].write_bitstream_my_mutex;
	}

	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		params[i].thread_num = i;

		ret = pthread_create(&thread[i], &attr, &thread_start_routin, (void*)&params[i]);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
//	printf("%d %p\n", params[0].seq, &params[0].write_bitstream_my_mutex);
	pthread_mutex_unlock(&params[0].write_bitstream_my_mutex);

	for(i=0;i<MAX_THREAD_NUMBER;i++) {
			pthread_join(thread[i], NULL);
	}
	printf("main end\n");
	return 0;

#if 0
	int ret;
	int thread_number = 0;
	int end_thread_number = 0;
	int seq_number = 0;




	for(seq_number=0;seq_number < MAX_SEQ_NUMBER;seq_number++) {

		params[seq_number % MAX_THREAD_NUMBER ].seq = seq_number; 
		pthread_mutex_unlock(&params[seq_number % MAX_THREAD_NUMBER ].my_thread_mutex);

		thread_number++;
		if (thread_number == MAX_THREAD_NUMBER) {
//			printf("join %d %d\n", thread_number, end_thread_number);
//			pthread_join(thread[end_thread_number], NULL);
			end_thread_number++;
			thread_number--;
			if (end_thread_number == MAX_SEQ_NUMBER) {
				break;
			}

		}


	}
	if (end_thread_number != MAX_SEQ_NUMBER) {
		for(;;) {
//			printf("join %d %d\n", thread_number, end_thread_number);
			pthread_join(thread[end_thread_number], NULL);
			end_thread_number++;
			thread_number--;
			if (end_thread_number == MAX_SEQ_NUMBER) {
				break;
			}
		}
	}
	printf("main end %d\n", thread_number);
	return 0;
#endif

}