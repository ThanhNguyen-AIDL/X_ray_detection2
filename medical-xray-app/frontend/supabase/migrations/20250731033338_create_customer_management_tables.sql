-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create customers table
CREATE TABLE customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    date_of_birth DATE,
    gender VARCHAR(10) CHECK (gender IN ('Nam', 'Nữ', 'Khác')),
    address TEXT,
    medical_history TEXT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create customer_images table
CREATE TABLE customer_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID REFERENCES customers(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    image_type VARCHAR(50) DEFAULT 'X-ray',
    body_part VARCHAR(100),
    description TEXT,
    file_name VARCHAR(255),
    file_size INTEGER,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Update xray_analyses table to link with customers and images
ALTER TABLE xray_analyses 
ADD COLUMN customer_id UUID REFERENCES customers(id) ON DELETE SET NULL,
ADD COLUMN image_id UUID REFERENCES customer_images(id) ON DELETE SET NULL,
ADD COLUMN body_part VARCHAR(100),
ADD COLUMN severity VARCHAR(20);

-- Create indexes for better performance
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_phone ON customers(phone);
CREATE INDEX idx_customers_created_at ON customers(created_at);
CREATE INDEX idx_customer_images_customer_id ON customer_images(customer_id);
CREATE INDEX idx_customer_images_uploaded_at ON customer_images(uploaded_at);
CREATE INDEX idx_xray_analyses_customer_id ON xray_analyses(customer_id);
CREATE INDEX idx_xray_analyses_image_id ON xray_analyses(image_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for customers table
CREATE TRIGGER update_customers_updated_at 
    BEFORE UPDATE ON customers 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO customers (full_name, email, phone, date_of_birth, gender, address, medical_history) VALUES
('Nguyễn Văn An', 'nguyenvanan@email.com', '0901234567', '1990-05-15', 'Nam', '123 Đường ABC, Quận 1, TP.HCM', 'Không có tiền sử bệnh đặc biệt'),
('Trần Thị Bình', 'tranthibinh@email.com', '0912345678', '1985-08-22', 'Nữ', '456 Đường XYZ, Quận 3, TP.HCM', 'Tiền sử viêm phổi năm 2020'),
('Lê Minh Cường', 'leminhcuong@email.com', '0923456789', '1995-12-10', 'Nam', '789 Đường DEF, Quận 7, TP.HCM', 'Gãy xương tay năm 2018');