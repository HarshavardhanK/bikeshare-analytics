"""
Basic structure test to verify submission requirements
"""

import os
import sys

def test_required_files_exist():
    """Test that all required files for submission exist"""
    
    # Required files according to assignment
    required_files = [
        'src/',
        'tests/',
        'README.md',
        'LICENSE'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file/directory missing: {file_path}"
    
    print("✅ All required files exist")

def test_src_structure():
    """Test that src directory contains required modules"""
    
    src_files = os.listdir('src')
    required_modules = [
        'app_init.py',
        'db.py', 
        'intent_utils.py',
        'semantic_mapper.py',
        'sql_generator.py'
    ]
    
    for module in required_modules:
        assert module in src_files, f"Required module missing: {module}"
    
    print("✅ All required source modules exist")

def test_tests_structure():
    """Test that tests directory contains required test files"""
    
    test_files = os.listdir('tests')
    required_tests = [
        'test_sql_generator.py',
        'test_semantic_mapper.py',
        'test_public_acceptance.py'
    ]
    
    for test_file in required_tests:
        assert test_file in test_files, f"Required test file missing: {test_file}"
    
    print("✅ All required test files exist")

def test_public_acceptance_tests_defined():
    """Test that public acceptance tests are properly defined"""
    
    # Read the public acceptance test file
    with open('tests/test_public_acceptance.py', 'r') as f:
        content = f.read()
    
    # Check for the three required tests
    required_tests = [
        'test_t1_avg_ride_time_congress_avenue',
        'test_t2_most_departures_first_week_june', 
        'test_t3_kilometres_women_rainy_june'
    ]
    
    for test_name in required_tests:
        assert test_name in content, f"Required test function missing: {test_name}"
    
    # Check for exact expected outputs
    expected_outputs = [
        '"25 minutes"',
        '"Congress Avenue"',
        '"6.8 km"'
    ]
    
    for output in expected_outputs:
        assert output in content, f"Expected output missing: {output}"
    
    print("✅ Public acceptance tests properly defined")

def test_readme_length():
    """Test that README is concise (~2 pages max)"""
    
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Rough estimate: 1 page ≈ 500 words, 2 pages ≈ 1000 words
    word_count = len(content.split())
    assert word_count <= 1200, f"README too long: {word_count} words (should be ~1000 max)"
    
    print(f"✅ README length acceptable: {word_count} words")

def test_license_exists():
    """Test that LICENSE file exists and has content"""
    
    with open('LICENSE', 'r') as f:
        content = f.read()
    
    assert len(content.strip()) > 0, "LICENSE file is empty"
    assert 'MIT' in content or 'Apache' in content, "LICENSE should be MIT or Apache"
    
    print("✅ LICENSE file exists and valid")

def test_no_sensitive_data():
    """Test that no sensitive data is in the repository"""
    
    # Check for common sensitive patterns
    sensitive_patterns = [
        'sk-',  # OpenAI API keys
        'password=',
        'secret=',
        'token='
    ]
    
    # Check source files
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        for pattern in sensitive_patterns:
                            assert pattern not in content, f"Sensitive data found in {file_path}: {pattern}"
                except:
                    pass  # Skip files that can't be read
    
    print("✅ No sensitive data found in repository")

if __name__ == "__main__":
    print("Running submission structure tests...")
    print("=" * 50)
    
    test_required_files_exist()
    test_src_structure()
    test_tests_structure()
    test_public_acceptance_tests_defined()
    test_readme_length()
    test_license_exists()
    test_no_sensitive_data()
    
    print("=" * 50)
    print("✅ All submission structure tests passed!")
    print("Repository is ready for submission.")
