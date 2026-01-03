#!/usr/bin/env python3
"""
BDD Workflow Example - Complete Behavior-Driven Development workflow.

This example demonstrates how to use code2logic to generate
BDD artifacts including Gherkin features and Cucumber configurations.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import ProjectAnalyzer
from code2logic.gherkin import GherkinGenerator, CucumberYAMLGenerator, generate_gherkin_from_project
from code2logic.models import create_project, create_module, create_function, create_class


def create_sample_bdd_project():
    """Create a sample project suitable for BDD demonstration."""
    print("🎭 Creating sample BDD project...")
    
    project = create_project(
        name="ecommerce_system",
        path="/tmp/ecommerce_system"
    )
    
    # User management module
    user_module = create_module(
        name="user_management",
        path="/tmp/ecommerce_system/user_management.py",
        functions=[
            create_function(
                name="register_user",
                parameters=["username", "email", "password"],
                lines_of_code=15,
                complexity=4,
                docstring="Register a new user account",
                code="def register_user(username, email, password):\n    # Validate input\n    # Check if user exists\n    # Hash password\n    # Save to database\n    return user_id"
            ),
            create_function(
                name="authenticate_user",
                parameters=["email", "password"],
                lines_of_code=10,
                complexity=3,
                docstring="Authenticate user credentials",
                code="def authenticate_user(email, password):\n    # Find user by email\n    # Verify password\n    # Return session token\n    return token"
            ),
            create_function(
                name="update_user_profile",
                parameters=["user_id", "profile_data"],
                lines_of_code=8,
                complexity=2,
                docstring="Update user profile information",
                code="def update_user_profile(user_id, profile_data):\n    # Validate data\n    # Update database\n    return success"
            )
        ],
        classes=[
            create_class(
                name="User",
                methods=[
                    create_function(
                        name="__init__",
                        parameters=["username", "email"],
                        lines_of_code=5,
                        complexity=1
                    ),
                    create_function(
                        name="validate_email",
                        parameters=[],
                        lines_of_code=3,
                        complexity=1
                    ),
                    create_function(
                        name="set_password",
                        parameters=["password"],
                        lines_of_code=4,
                        complexity=2
                    )
                ],
                lines_of_code=20,
                docstring="User entity class"
            )
        ],
        imports=["hashlib", "re", "datetime"],
        lines_of_code=50
    )
    
    # Product management module
    product_module = create_module(
        name="product_management",
        path="/tmp/ecommerce_system/product_management.py",
        functions=[
            create_function(
                name="create_product",
                parameters=["name", "price", "description"],
                lines_of_code=12,
                complexity=3,
                docstring="Create a new product",
                code="def create_product(name, price, description):\n    # Validate product data\n    # Generate SKU\n    # Save to database\n    return product_id"
            ),
            create_function(
                name="update_inventory",
                parameters=["product_id", "quantity"],
                lines_of_code=8,
                complexity=2,
                docstring="Update product inventory",
                code="def update_inventory(product_id, quantity):\n    # Check product exists\n    # Update inventory\n    return new_quantity"
            ),
            create_function(
                name="search_products",
                parameters=["query", "filters"],
                lines_of_code=15,
                complexity=5,
                docstring="Search products with filters",
                code="def search_products(query, filters):\n    # Build search query\n    # Apply filters\n    # Execute search\n    # Return results\n    return products"
            )
        ],
        classes=[
            create_class(
                name="Product",
                methods=[
                    create_function(
                        name="__init__",
                        parameters=["name", "price"],
                        lines_of_code=4,
                        complexity=1
                    ),
                    create_function(
                        name="calculate_discount",
                        parameters=["discount_rate"],
                        lines_of_code=6,
                        complexity=2
                    ),
                    create_function(
                        name="is_available",
                        parameters=[],
                        lines_of_code=3,
                        complexity=1
                    )
                ],
                lines_of_code=18,
                docstring="Product entity class"
            )
        ],
        imports=["datetime", "decimal"],
        lines_of_code=45
    )
    
    # Order processing module
    order_module = create_module(
        name="order_processing",
        path="/tmp/ecommerce_system/order_processing.py",
        functions=[
            create_function(
                name="create_order",
                parameters=["user_id", "items", "shipping_address"],
                lines_of_code=20,
                complexity=6,
                docstring="Create a new order",
                code="def create_order(user_id, items, shipping_address):\n    # Validate user\n    # Check inventory\n    # Calculate total\n    # Apply discounts\n    # Create order record\n    return order_id"
            ),
            create_function(
                name="process_payment",
                parameters=["order_id", "payment_method"],
                lines_of_code=15,
                complexity=4,
                docstring="Process payment for order",
                code="def process_payment(order_id, payment_method):\n    # Validate payment method\n    # Process transaction\n    # Update order status\n    return payment_status"
            ),
            create_function(
                name="ship_order",
                parameters=["order_id", "carrier"],
                lines_of_code=10,
                complexity=3,
                docstring="Ship order to customer",
                code="def ship_order(order_id, carrier):\n    # Validate order status\n    # Create shipment\n    # Update tracking\n    return tracking_number"
            )
        ],
        classes=[
            create_class(
                name="Order",
                methods=[
                    create_function(
                        name="__init__",
                        parameters=["user_id"],
                        lines_of_code=4,
                        complexity=1
                    ),
                    create_function(
                        name="add_item",
                        parameters=["product", "quantity"],
                        lines_of_code=5,
                        complexity=2
                    ),
                    create_function(
                        name="calculate_total",
                        parameters=[],
                        lines_of_code=8,
                        complexity=3
                    ),
                    create_function(
                        name="get_status",
                        parameters=[],
                        lines_of_code=3,
                        complexity=1
                    )
                ],
                lines_of_code=25,
                docstring="Order entity class"
            )
        ],
        imports=["datetime", "decimal", "uuid"],
        lines_of_code=60
    )
    
    project.modules.extend([user_module, product_module, order_module])
    
    print(f"✅ Created BDD project with {len(project.modules)} modules")
    return project


def demonstrate_gherkin_generation(project):
    """Demonstrate Gherkin feature generation."""
    print("\n🥒 Generating Gherkin features...")
    
    # Create output directory
    output_dir = Path("./bdd_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Gherkin features
    gherkin_gen = GherkinGenerator()
    features = gherkin_gen.generate_features(project)
    
    print(f"📋 Generated {len(features)} Gherkin features:")
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature.name} ({len(feature.scenarios)} scenarios)")
        
        # Write feature files
        gherkin_gen.write_feature_files(str(output_dir / "features"))
    
    # Show sample feature content
    if features:
        sample_feature = features[0]
        print(f"\n📄 Sample Feature: {sample_feature.name}")
        print("-" * 40)
        feature_content = gherkin_gen._format_feature(sample_feature)
        lines = feature_content.split('\n')
        for line in lines[:20]:  # Show first 20 lines
            print(line)
        if len(lines) > 20:
            print("... (truncated)")
    
    return features


def demonstrate_cucumber_config(project):
    """Demonstrate Cucumber YAML configuration."""
    print("\n🥒 Generating Cucumber configuration...")
    
    # Create output directory
    output_dir = Path("./bdd_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate Cucumber configuration
    yaml_gen = CucumberYAMLGenerator()
    config = yaml_gen.generate_cucumber_config(project)
    
    # Write configuration
    config_path = output_dir / "cucumber.yml"
    yaml_gen.write_cucumber_config(str(config_path), config)
    
    print(f"✅ Cucumber configuration saved to: {config_path}")
    
    # Show key configuration sections
    print("\n📋 Configuration Summary:")
    print(f"   Step definitions: {len(config['step_definitions'])}")
    print(f"   Feature paths: {len(config['features'])}")
    print(f"   Glue paths: {len(config['glue'])}")
    print(f"   Output formats: {', '.join(config['format'][:3])}...")
    
    return config


def demonstrate_step_definitions(project):
    """Demonstrate step definition generation."""
    print("\n👟 Generating step definitions...")
    
    # Create output directory
    output_dir = Path("./bdd_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate step definitions
    yaml_gen = CucumberYAMLGenerator()
    step_defs = yaml_gen.generate_step_definitions(project)
    
    # Write step definitions
    steps_dir = output_dir / "step_definitions"
    steps_dir.mkdir(exist_ok=True)
    
    for step_type, code in step_defs.items():
        step_file = steps_dir / f"{step_type}.rb"
        with open(step_file, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"   📄 {step_file}")
    
    # Show sample step definitions
    print("\n📝 Sample Step Definitions:")
    print("-" * 30)
    given_steps = step_defs.get("given_steps", "")
    lines = given_steps.split('\n')
    for line in lines[:10]:  # Show first 10 lines
        if line.strip():
            print(line)
    if len(lines) > 10:
        print("... (truncated)")
    
    return step_defs


def generate_bdd_report(project, features, config, step_defs):
    """Generate a comprehensive BDD report."""
    print("\n📊 Generating BDD report...")
    
    output_dir = Path("./bdd_output")
    
    # Create report data
    report = {
        "project": {
            "name": project.name,
            "modules": len(project.modules),
            "functions": sum(len(m.functions) for m in project.modules),
            "classes": sum(len(m.classes) for m in project.modules),
            "lines_of_code": sum(m.lines_of_code for m in project.modules)
        },
        "bdd_artifacts": {
            "gherkin_features": len(features),
            "total_scenarios": sum(len(f.scenarios) for f in features),
            "cucumber_config": bool(config),
            "step_definition_types": len(step_defs)
        },
        "features": [
            {
                "name": f.name,
                "scenarios": len(f.scenarios),
                "description": f.description
            }
            for f in features
        ],
        "modules": [
            {
                "name": m.name,
                "functions": len(m.functions),
                "classes": len(m.classes),
                "complexity": sum(f.complexity for f in m.functions)
            }
            for m in project.modules
        ]
    }
    
    # Write JSON report
    report_path = output_dir / "bdd_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ BDD report saved to: {report_path}")
    
    # Print summary
    print(f"\n📈 BDD Summary:")
    print(f"   Project: {report['project']['name']}")
    print(f"   Modules: {report['project']['modules']}")
    print(f"   Functions: {report['project']['functions']}")
    print(f"   Classes: {report['project']['classes']}")
    print(f"   Gherkin Features: {report['bdd_artifacts']['gherkin_features']}")
    print(f"   Total Scenarios: {report['bdd_artifacts']['total_scenarios']}")
    
    return report


def show_bdd_workflow_summary():
    """Show BDD workflow summary and next steps."""
    print("\n🎯 BDD Workflow Summary")
    print("=" * 40)
    print("What we accomplished:")
    print("✅ Created a sample e-commerce project")
    print("✅ Generated Gherkin features for all modules")
    print("✅ Created Cucumber configuration")
    print("✅ Generated step definitions")
    print("✅ Created comprehensive BDD report")
    print()
    print("Generated files:")
    print("📁 bdd_output/")
    print("   ├── features/           # Gherkin feature files")
    print("   ├── step_definitions/  # Ruby step definitions")
    print("   ├── cucumber.yml       # Cucumber configuration")
    print("   └── bdd_report.json    # Complete BDD report")
    print()
    print("Next steps:")
    print("🧪 Run Cucumber tests: cucumber -p default")
    print("🔧 Customize step definitions for your specific logic")
    print("📝 Add more detailed scenarios to features")
    print("🚀 Integrate with CI/CD pipeline")
    print("📊 Use BDD report for project documentation")


def main():
    """Main BDD workflow demonstration."""
    print("🎭 BDD Workflow Example")
    print("=" * 50)
    print("This example demonstrates a complete BDD workflow")
    print("using code2logic to generate Gherkin features and Cucumber configs.")
    print()
    
    try:
        # Step 1: Create sample project
        project = create_sample_bdd_project()
        
        # Step 2: Generate Gherkin features
        features = demonstrate_gherkin_generation(project)
        
        # Step 3: Generate Cucumber configuration
        config = demonstrate_cucumber_config(project)
        
        # Step 4: Generate step definitions
        step_defs = demonstrate_step_definitions(project)
        
        # Step 5: Generate BDD report
        report = generate_bdd_report(project, features, config, step_defs)
        
        # Step 6: Show workflow summary
        show_bdd_workflow_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is a demonstration - some errors are expected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
