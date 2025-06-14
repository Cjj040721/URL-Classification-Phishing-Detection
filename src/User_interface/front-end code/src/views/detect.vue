<template>
  <div class="container">
    <div class="left-section">
      <!-- Input Section -->
      <div class="input-section">
        <el-input
          v-model="urlInput"
          placeholder="Enter URL to check"
          class="url-input"
          @keyup.enter="checkUrl"
        >
          <template #append>
            <el-button :loading="loading" @click="checkUrl" type="primary">
              Check URL
            </el-button>
          </template>
        </el-input>
      </div>

      <!-- Statistics Cards -->
      <div class="stats-container">
        <el-card class="stat-card">
          <div class="stat-title">Total URLs</div>
          <div class="stat-value">{{ stats.total }}</div>
        </el-card>
        <el-card class="stat-card safe">
          <div class="stat-title">Safe Websites</div>
          <div class="stat-value">{{ stats.safe }}</div>
        </el-card>
        <el-card class="stat-card phishing">
          <div class="stat-title">Phishing Websites</div>
          <div class="stat-value">{{ stats.phishing }}</div>
        </el-card>
      </div>

      <!-- Charts -->
      <div class="chart-container">
        <v-chart class="chart" :option="chartOption" autoresize />
      </div>
    </div>

    <!-- History Table -->
    <div class="right-section">
      <el-table :data="history" class="history-table">
        <el-table-column prop="url" label="URL" width="300" />
        <el-table-column label="Status" width="120">
          <template #default="scope">
            <el-tag 
              v-if="scope.row"
              :type="scope.row.isPhishing ? 'danger' : 'success'"
              effect="dark"
            >
              {{ scope.row.isPhishing ? 'Phishing' : 'Safe' }}
            </el-tag>
            <span v-else>N/A</span>
          </template>
        </el-table-column>
        <el-table-column prop="timestamp" label="Checked At" />
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { ElInput, ElButton, ElAlert, ElRow, ElCol, ElTable, ElTableColumn } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
  CanvasRenderer,
  PieChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

const urlInput = ref('')
const loading = ref(false)
const history = ref([])

const stats = reactive({
  total: 0,
  safe: 0,
  phishing: 0
})

const chartOption = computed(() => ({
  title: {
    text: 'Website Safety Distribution',
    left: 'center'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
    orient: 'vertical',
    left: 'left'
  },
  series: [
    {
      name: 'URL Status',
      type: 'pie',
      radius: '50%',
      data: [
        { value: stats.safe, name: 'Safe Websites' },
        { value: stats.phishing, name: 'Phishing Websites' }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
}))

async function checkUrl() {
  if (!urlInput.value) {
    ElMessage.warning('Please enter a URL')
    return
  }

  try {
    loading.value = true
    // Replace with actual API call
    const response = await fetch('http://localhost:5000/detect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: urlInput.value })
    })

    const result = await response.json()
    const newEntry = {
      url: urlInput.value,
      isPhishing: result.isPhishing,
      timestamp: new Date().toLocaleString()
    }

    history.value.unshift(newEntry)
    stats.total++
    result.isPhishing ? stats.phishing++ : stats.safe++
    urlInput.value = ''
  } catch (error) {
    ElMessage.error('Error checking URL')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  display: flex;
  gap: 2rem;
}

.left-section {
  flex: 1;
}

.right-section {
  width: 800px;
}

.input-section {
  margin-bottom: 2rem;
}

.url-input {
  max-width: 600px;
  margin: 0 auto;
}

.stats-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  text-align: center;
  padding: 1.5rem;
}

.stat-title {
  font-size: 1.1rem;
  color: #666;
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: bold;
}

.safe {
  border-top: 4px solid #67c23a;
}

.phishing {
  border-top: 4px solid #f56c6c;
}

.chart-container {
  height: 400px;
  margin-bottom: 2rem;
}

.history-table {
  height: calc(100vh - 4rem);
  overflow-y: auto;
}
</style>