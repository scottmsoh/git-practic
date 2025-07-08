pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('check_change_target'){
            steps {
                echo "Build complete for ${env.CHANGE_TARGET}"
            }
        }
        stage('Test for PR to develop') {
            when {
                allOf {
                    expression { env.CHANGE_ID != null }
                    expression { env.CHANGE_TARGET == 'develop' }
                }
            }
            steps {
                sh 'python3 ./test/test.py'
            }
        }
    }
}
